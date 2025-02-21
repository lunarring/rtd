import torch
from rtd.dynamic_processor.base_dynamic_module import BaseDynamicClass

class DynamicClass(BaseDynamicClass):
    def __init__(self):
        super().__init__()

    def process(
        self,
        img_camera: torch.Tensor,
        img_mask_segmentation: torch.Tensor,
        img_diffusion: torch.Tensor,
        img_optical_flow: torch.Tensor,
        list_dynamic_coef: list,
    ) -> torch.Tensor:
        # Unpack dynamic coefficients
        a = list_dynamic_coef[0]
        b = list_dynamic_coef[1]
        c = list_dynamic_coef[2]

        # Apply the segmentation mask to the camera image
        masked_camera = img_camera * img_mask_segmentation

        # Compute a flow component by averaging across the last dimension.
        flow_component = torch.mean(img_optical_flow, dim=-1, keepdim=True)
        if flow_component.shape[-1] != img_camera.shape[-1]:
            flow_component = flow_component.expand_as(img_camera)

        # Combine images using the dynamic coefficients.
        # This uses masked_camera, diffusion, and the averaged optical flow.
        output = a * masked_camera + b * img_diffusion + c * flow_component

        # Ensure the output values remain within the valid range
        output = torch.clamp(output, 0, 255)
        return output.float()