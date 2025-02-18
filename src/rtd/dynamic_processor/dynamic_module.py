import torch
from rtd.dynamic_processor.base_dynamic_module import BaseDynamicClass

class DynamicClass(BaseDynamicClass):
    def process(
        self,
        img_camera: torch.Tensor,
        img_mask_segmentation: torch.Tensor,
        img_diffusion: torch.Tensor,
        img_optical_flow: torch.Tensor,
        dynamic_func_coef: float,
    ) -> torch.Tensor:
        clamped_img = torch.clamp(img_camera, 0, 255)
        return clamped_img
