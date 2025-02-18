import torch
from rtd.dynamic_processor.base_dynamic_module import BaseDynamicClass

class DynamicClass(BaseDynamicClass):
    def process(
        self,
        img_camera: torch.Tensor,
        img_mask_segmentation: torch.Tensor,
        img_diffusion: torch.Tensor,
        dynamic_func_coef: float,
    ) -> torch.Tensor:
        # Multiply the camera image by the segmentation mask so that masked areas become zero.
        result = img_camera * img_mask_segmentation
        # Clamp the output to ensure values are in the valid range [0, 255]
        result = torch.clamp(result, 0, 255)
        return result
