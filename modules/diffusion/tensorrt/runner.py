import gc
import logging
import json
import os
import random
import time
from typing import Optional, Tuple

import diffusers
import numpy as np
import tensorrt as trt
import torch
from PIL import Image
from polygraphy import cuda
from tqdm import tqdm
from transformers import CLIPTokenizer

from api.events.generation import PreInferenceEvent, PreLatentsCreateEvent
from api.generation import (
    ImageGenerationOptions,
    ImageGenerationResult,
    ImageInformation,
)
from api.generation.image_generation_response import ImageGenerationProgress
from lib.trt.utilities import TRT_LOGGER, Engine
from lib.trt.utilities import DPMScheduler, DDIMScheduler, EulerAncestralDiscreteScheduler, LMSDiscreteScheduler, PNDMScheduler
from modules import utils

from ..runner import BaseRunner
from . import clip
from .models import CLIP, VAE, UNet
from .pwp import TensorRTPromptWeightingPipeline
from .upscaler import Upscaler

log_level = os.environ.get('LOG_LEVEL', 'INFO')
logging.basicConfig(level=getattr(logging, log_level),
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger()


def preprocess_image(image: Image.Image, height: int, width: int):
    width, height = map(lambda x: x - x % 8, (width, height))
    image = image.resize(
        (width, height), resample=diffusers.utils.PIL_INTERPOLATION["lanczos"]
    )
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


class TensorRTDiffusionRunner(BaseRunner):
    def __init__(self, model_dir: str):
        meta_filepath = os.path.join(model_dir, "model_index.json")
        assert os.path.exists(meta_filepath), "Model meta data not found."
        engine_dir = os.path.join(model_dir, "engine")

        with open(meta_filepath, mode="r") as f:
            txt = f.read()
            self.meta = json.loads(txt)

        self.upscaler = Upscaler()
        self.scheduler = None
        self.scheduler_id = None
        self.model_id = self.meta.get("model_id","CompVis/stable-diffusion-v1-4")
        self.device = torch.device("cuda")
        self.fp16 = self.meta["denoising_prec"] == "fp16"
        self.engines = {
            "clip": clip.create_clip_engine(),
            "unet": Engine("unet", engine_dir),
            "vae": Engine("vae", engine_dir),
        }
        self.models = {
            "clip": CLIP(clip.model_id, fp16=self.fp16, device=self.device),
            "unet": UNet(
                self.meta["models"]["unet"],
                subfolder=self.meta["subfolder"],
                fp16=self.fp16,
                device=self.device,
            ),
            "vae": VAE(
                self.meta["models"]["vae"],
                subfolder=self.meta["subfolder"],
                fp16=self.fp16,
                device=self.device,
            ),
        }
        #self.en_vae = self.models["vae"].get_model()

        for model in self.models.values():
            model.min_latent_shape = self.meta.get("min_latent_resolution", 256) // 8
            model.max_latent_shape = self.meta.get("max_latent_resolution", 1024) // 8

    def activate(
        self,
        tokenizer_id="openai/clip-vit-large-patch14",
    ):
        self.stream = cuda.Stream()
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_id)
        self.pwp = TensorRTPromptWeightingPipeline(
            tokenizer=self.tokenizer,
            text_encoder=self.engines["clip"],
            stream=self.stream,
            device=self.device,
        )

        for engine in self.engines.values():
            engine.activate()
        self.loading = False

    def teardown(self):
        for engine in self.engines.values():
            del engine
        self.stream.free()
        del self.stream
        del self.tokenizer
        del self.pwp.text_encoder
        torch.cuda.empty_cache()
        gc.collect()

    def loadResources(self, image_height, image_width, batch_size, seed):
        self.generator = torch.Generator(device="cuda").manual_seed(seed)
        # Allocate buffers for TensorRT engine bindings
        for model_name, obj in self.models.items():
            self.engines[model_name].allocate_buffers(shape_dict=obj.get_shape_dict(batch_size, image_height, image_width), device=self.device)

    def run_engine(self, model_name, feed_dict):
        engine = self.engines[model_name]
        return engine.infer(feed_dict, self.stream)

    def get_scheduler(self, scheduler_id: str):
        sched_opts = {
            'num_train_timesteps': 1000,
            'beta_start': 0.00085,
            'beta_end': 0.012
        }
        if scheduler_id == "ddim":
            return DDIMScheduler(device=self.device, **sched_opts)
        elif scheduler_id == "dpm++":
            return DPMScheduler(device=self.device, **sched_opts)
        elif scheduler_id == "euler_a":
            return EulerAncestralDiscreteScheduler(device=self.device, **sched_opts)
        elif scheduler_id == "lmsd":
            return LMSDiscreteScheduler(device=self.device, **sched_opts)
        elif scheduler_id == "pndm":
            sched_opts["steps_offset"] = 1
            return PNDMScheduler(device=self.device, **sched_opts)
        else:
            raise ValueError(f"Scheduler should be either ddim, dpm++, euler_a, lmsd or pndm")

    def initialize_latents(self, batch_size, unet_channels, latent_height, latent_width):
        latents_dtype = torch.float32 # text_embeddings.dtype
        latents_shape = (batch_size, unet_channels, latent_height, latent_width)
        latents = torch.randn(latents_shape, device=self.device, dtype=latents_dtype, generator=self.generator)

        # Scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def initialize_timesteps(self, timesteps, strength, is_txt2img):
        self.scheduler.set_timesteps(timesteps)
        self.scheduler.configure()
        if is_txt2img:
            return self.scheduler.timesteps, timesteps
        else:
            offset = self.scheduler.steps_offset if hasattr(self.scheduler, "steps_offset") else 0
            init_timestep = int(timesteps * strength) + offset
            init_timestep = min(init_timestep, timesteps)
            t_start = max(timesteps - init_timestep + offset, 0)
            timesteps = self.scheduler.timesteps[t_start:].to(self.device)
            return timesteps, t_start

    def denoise_latent(self, latents, text_embeddings, timesteps=None, step_offset=0, mask=None, masked_image_latents=None):
        if not isinstance(timesteps, torch.Tensor):
            timesteps = self.scheduler.timesteps

        for step_index, timestep in enumerate(timesteps):
            # Expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, step_offset + step_index, timestep)
            if isinstance(mask, torch.Tensor):
                latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)

            timestep_float = timestep.float() if timestep.dtype != torch.float32 else timestep

            sample_inp = cuda.DeviceView(
                ptr=latent_model_input.data_ptr(),
                shape=latent_model_input.shape,
                dtype=np.float32,
            )
            timestep_inp = cuda.DeviceView(
                ptr=timestep_float.data_ptr(),
                shape=timestep_float.shape,
                dtype=np.float32,
            )
            embeddings_inp = cuda.DeviceView(
                ptr=text_embeddings.data_ptr(),
                shape=text_embeddings.shape,
                dtype=np.float16 if self.fp16 else np.float32,
            )
            noise_pred = self.run_engine(
                "unet",
                {
                    "sample": sample_inp,
                    "timestep": timestep_inp,
                    "encoder_hidden_states": embeddings_inp,
                },
            )["latent"]

            # Perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
            latents = self.scheduler.step(noise_pred, latents, step_offset + step_index, timestep)

        latents = 1. / 0.18215 * latents
        return latents

    def decode_latent(self, latents):
        sample_inp = cuda.DeviceView(
            ptr=latents.data_ptr(), shape=latents.shape, dtype=np.float32
        )
        images = self.run_engine("vae", {"latent": sample_inp})["images"]
        return images

    def to_image(self, images, upscale=False):
        images = (
            ((images + 1) * 255 / 2)
            .clamp(0, 255)
            .detach()
            .permute(0, 2, 3, 1)
            .round()
            .type(torch.uint8)
            .cpu()
            .numpy()
        )
        result = []
        for i in range(images.shape[0]):
            image = images[i]

            if upscale:
                image = self.upscaler.upscale(image, outscale=2)

            image = Image.fromarray(image)
            result.append(image)
        return result

    def infer(self, opts: ImageGenerationOptions):
        self.wait_loading()
        if opts.img is not None:
            opts.img = utils.b642img(opts.img)
            opts.img = preprocess_image(
                opts.img, opts.image_height, opts.image_width
            ).to(device=self.device)
        pre_inference_event = PreInferenceEvent(opts)
        PreInferenceEvent.call_event(pre_inference_event)

        self.guidance_scale = opts.scale

        # Set scheduler.
        if self.scheduler_id != opts.scheduler_id:
            self.scheduler = self.get_scheduler(opts.scheduler_id)

        # Pre-compute latent input scales and linear multistep coefficients
        timesteps, steps = self.initialize_timesteps(opts.steps, opts.strength, opts.img is None)
        latent_timestep = timesteps[:1].repeat(opts.batch_size * opts.batch_count)

        results = []

        if opts.seed is None or opts.seed == -1:
            opts.seed = random.randrange(0, 4294967294, 1)

        torch.cuda.synchronize()
        e2e_tic = time.perf_counter()

        # Start pipeline
        for i in range(opts.batch_count):
            manual_seed = opts.seed + i
            self.loadResources(opts.image_height, opts.image_width, opts.batch_size, manual_seed)

            with torch.inference_mode(), torch.autocast("cuda"), trt.Runtime(
                TRT_LOGGER
            ):
                # Text encoder
                text_embeddings = self.pwp(
                    prompt=opts.prompt,
                    negative_prompt=opts.negative_prompt,
                    guidance_scale=opts.scale,
                    batch_size=opts.batch_size,
                    max_embeddings_multiples=1,
                )
                if self.fp16:
                    text_embeddings = text_embeddings.to(dtype=torch.float16)

                # Prepare latent space
                pre_latents_create_event = PreLatentsCreateEvent(opts)
                PreLatentsCreateEvent.call_event(pre_latents_create_event)

                latents = pre_latents_create_event.latents

                if not pre_latents_create_event.skip:
                    latents = self.initialize_latents(
                        batch_size=opts.batch_size,
                        unet_channels=4,
                        latent_height=(opts.image_height // 8),
                        latent_width=(opts.image_width // 8)
                    )

                # UNET denoiser
                latents = self.denoise_latent(latents, text_embeddings)

                # Decode latent with VAE.
                images = self.decode_latent(latents)

                torch.cuda.synchronize()

                # Image metadata.
                info = ImageInformation(
                    prompt=opts.prompt,
                    negative_prompt=opts.negative_prompt,
                    steps=opts.steps,
                    scale=opts.scale,
                    seed=manual_seed,
                    height=opts.image_height,
                    width=opts.image_width,
                    img2img=opts.img is not None,
                    strength=opts.strength,
                )

                if opts.generator:
                    yield ImageGenerationResult(
                        images={utils.img2b64(x) : info for x in self.to_image(images)},
                        performance=time.perf_counter() - e2e_tic,
                    )
                else:
                    results.append((self.to_image(images), info))
        if not opts.generator:
            all_perf_time = time.perf_counter() - e2e_tic

            result = ImageGenerationResult(images={}, performance=all_perf_time)
            for x in results:
                (images, info) = x
                for img in images:
                    result.images[utils.img2b64(img)] = info
            yield result
