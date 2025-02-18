import torch
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
        # Handle case when optical flow is omitted (4-argument call)
        if isinstance(img_optical_flow, float):
            dynamic_func_coef = img_optical_flow
            img_optical_flow = None

        if img_optical_flow is None:
            # Blend camera and diffusion images based on dynamic coefficient.
            blended = img_camera * (1 - dynamic_func_coef) + img_diffusion * dynamic_func_coef
        else:
            # Compute magnitude of optical flow.
            flow_mag = torch.sqrt(torch.pow(img_optical_flow[..., 0], 2) + torch.pow(img_optical_flow[..., 1], 2))
            # Expand flow magnitude to 3 channels.
            flow_mag = flow_mag.unsqueeze(-1).expand_as(img_camera)
            # Creatively combine the images:
            blended = (img_camera * (1 - dynamic_func_coef) + (img_diffusion + flow_mag) * dynamic_func_coef) / 2.0

        return torch.clamp(blended, 0, 255)
