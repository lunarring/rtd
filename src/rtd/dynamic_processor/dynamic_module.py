from rtd.dynamic_processor.base_dynamic_module import BaseDynamicClass
import torch
import torch.nn.functional as F

class DynamicClass(BaseDynamicClass):
    def __init__(self):
        super().__init__()
        self.accumulated_flow = None

    def process(self,
                img_camera: torch.Tensor,
                img_mask_segmentation: torch.Tensor,
                img_diffusion: torch.Tensor,
                img_optical_flow: torch.Tensor,
                dynamic_func_coef: float) -> torch.Tensor:
        # Accumulate optical flow over time using a persistent variable:
        if self.accumulated_flow is None:
            self.accumulated_flow = img_optical_flow.clone()
        else:
            self.accumulated_flow = self.accumulated_flow + img_optical_flow

        # Prepare the camera image: from (H, W, C) to (1, C, H, W)
        cam = img_camera.permute(2, 0, 1).unsqueeze(0)
        H, W = img_camera.shape[0], img_camera.shape[1]

        # Create a normalized grid for sampling.
        # grid_sample expects normalized coordinates in range [-1,1]
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, steps=H, device=img_camera.device),
            torch.linspace(-1, 1, steps=W, device=img_camera.device),
            indexing="ij"
        )
        grid = torch.stack((x, y), dim=-1)  # Shape: (H, W, 2)

        # Use only the first two channels of accumulated_flow as displacement.
        flow = self.accumulated_flow[:, :, :2]
        norm_flow_x = flow[..., 0] * 2.0 / (W - 1)
        norm_flow_y = flow[..., 1] * 2.0 / (H - 1)
        norm_disp = torch.stack((norm_flow_x, norm_flow_y), dim=-1)
        
        # Create final sampling grid:
        sampling_grid = grid + norm_disp
        sampling_grid = sampling_grid.unsqueeze(0)  # Add batch dimension (1, H, W, 2)

        # Resample the camera image using grid_sample:
        displaced_cam = F.grid_sample(cam, sampling_grid, mode='bilinear', padding_mode='border', align_corners=True)
        displaced_cam = displaced_cam.squeeze(0).permute(1, 2, 0)  # Back to (H, W, C)

        # Remix the displaced camera image with the diffusion image using the dynamic coefficient:
        result = dynamic_func_coef * displaced_cam + (1 - dynamic_func_coef) * img_diffusion
        
        # Ensure output values are within the valid image range
        result = torch.clamp(result, 0, 255)
        return result.to(torch.float32)
