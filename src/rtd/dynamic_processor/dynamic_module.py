import torch
import torch.nn.functional as F
from rtd.dynamic_processor.base_dynamic_module import BaseDynamicClass

class DynamicClass(BaseDynamicClass):

    def process(
        self,
        img_camera: torch.Tensor,
        img_mask_segmentation: torch.Tensor,
        img_diffusion: torch.Tensor,
        img_optical_flow: torch.Tensor = None,
        dynamic_func_coef: float = 0.0,
    ) -> torch.Tensor:
        # Apply segmentation mask to camera image.
        if img_mask_segmentation.dim() == 2:
            img_mask_segmentation = img_mask_segmentation.unsqueeze(2)
        result = img_camera * (1 - img_mask_segmentation)
        return torch.clamp(result, 0, 255)
