import torch
import cv2

from .patch_gen import PatchGenerator
from models.utils import create_generator_val


class NetEngine:

    def __init__(self, engine="pt", GNet=None, model_path=None, device=torch.device('cpu')):
        """
        engine: 1. pt(pytorch), 2. snpe, etc
        GNet: used for pytorch engine, edcit from yaml
        model_path:
        device:
        """
        self.engine = engine
        if self.engine == "pt":
            self.net = create_generator_val(GNet, model_path, force_load=True).to(device)
            self.net.eval()
            self.device = device
    

    def forward(self, in_short_img, in_long_img, deblur_out=None):
        """
        iuput:
        in_short_img: short image, numpy array(NHWC), should be preprocessed.
        in_long_img: long image, numpy array(NHWC), should be preprocessed.
        deblur_out: deblur image, numpy array(NHWC), should be preprocessed.

        return:
        ret: list or numpy array(NHWC)
        """
        if self.engine == "pt":
            in_short_img = torch.from_numpy(in_short_img).permute(0, 3, 1, 2).contiguous().to(self.device)
            in_long_img = torch.from_numpy(in_long_img).permute(0, 3, 1, 2).contiguous().to(self.device)
            if deblur_out is not None:
                deblur_out = torch.from_numpy(deblur_out).permute(0, 3, 1, 2).contiguous().to(self.device)

            with torch.no_grad():
                if deblur_out is not None:
                    outs = self.net(in_short_img, in_long_img, deblur_out)
                else:
                    outs = self.net(in_short_img, in_long_img)

            if isinstance(outs, list):
                ret = []
                for i in range(len(outs)):
                    if len(outs[i].shape) == 4:  # suppose NCHW
                        out = outs[i].detach().cpu().permute(0, 2, 3, 1).numpy()
                        ret.append(out)
                    else:
                        ret.append(out)
            else:
                ret = outs.detach().cpu().permute(0, 2, 3, 1).numpy()
            return ret
