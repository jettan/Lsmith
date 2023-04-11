import os
import cv2

from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact


class Upscaler:
    def __init__(
            self,
            netscale=4,
            model_name="realesr-general-x4v3",
            denoise_strength=0.1,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            use_fp16=True
    ):
        if model_name == 'RealESRGAN_x4plus':  # x4 RRDBNet model
            self.model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
        elif model_name == 'RealESRGAN_x4plus_anime_6B':  # x4 RRDBNet model with 6 blocks
            self.model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
            file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth']
        else:
            #model_name == 'realesr-general-x4v3':  # x4 VGG-style model (S size)
            self.model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
            file_url = [
                    'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
                    'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
            ]

        self.model_path = os.path.join('models', model_name + '.pth')
        if not os.path.isfile(self.model_path):
            ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
            for url in file_url:
                self.model_path = load_file_from_url(
                    url=url, model_dir=os.path.join(ROOT_DIR, 'models'), progress=True, file_name=None)

        self.dni_weight=None
        if model_name == 'realesr-general-x4v3' and denoise_strength != 1:
            wdn_model_path = self.model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
            self.model_path = [self.model_path, wdn_model_path]
            self.dni_weight = [denoise_strength, 1 - denoise_strength]

        self.upscaler = RealESRGANer(
            scale=netscale,
            model_path=self.model_path,
            dni_weight=self.dni_weight,
            model=self.model,
            tile=tile,
            tile_pad=tile_pad,
            pre_pad=pre_pad,
            half=use_fp16,
            gpu_id=0,
        )

        self.model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)

    def upscale(self, img, outscale=4):
        output, _ = self.upscaler.enhance(img, outscale=outscale)
        return output
