from rtd.dynamic_processor.base_dynamic_module import BaseDynamicClass
import torch

class DynamicClass(BaseDynamicClass):
    def process(self, img_camera: torch.Tensor, img_mask_segmentation: torch.Tensor, img_diffusion: torch.Tensor, *args) -> torch.Tensor:
        # Support both signatures:
        # If only one additional argument, it's dynamic_func_coef,
        # If two, then the first extra is optical flow (unused) and second is dynamic_func_coef.
        if len(args) == 1:
            dynamic_func_coef = args[0]
        elif len(args) == 2:
            img_optical_flow, dynamic_func_coef = args
        else:
            raise ValueError("Invalid number of arguments passed to process")

        # Segment the image: only keep the parts where the segmentation mask is 1 (people)
        result = img_camera * img_mask_segmentation
        
        # Make sure output is clamped within the valid image range [0, 255]
        result = torch.clamp(result, 0, 255)
        return result.to(torch.float32)
