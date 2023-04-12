import os
from api.tensorrt import BuildEngineOptions

from modules import config

from .engine import build_engine
from .models import CLIP

model_id = "openai/clip-vit-large-patch14"


def create_clip_engine():
    model_data = CLIP(
        model_id,
        device="cuda",
        verbose=False,
        max_batch_size=16,
    )
    model_dir = os.path.join(config.get("model_dir"), model_id.replace("/", os.sep))
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(os.path.join(model_dir, "engine"), exist_ok=True)
    os.makedirs(os.path.join(model_dir, "onnx"), exist_ok=True)
    engine = build_engine(
        "clip",
        os.path.join(model_dir, "engine"),
        os.path.join(model_dir, "onnx"),
        model_data,
        BuildEngineOptions(model_id=model_id),
    )
    return engine
