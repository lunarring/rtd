import torch
import torch.nn.functional as F
from rtd.dynamic_processor.base_dynamic_module import BaseDynamicClass


class DynamicClass(BaseDynamicClass):
    def __init__(self):
        super().__init__()
        self.buffer_size = 10
        self.optical_flow_buffer = []

    def process(
        self,
        img_camera: torch.Tensor,
        img_mask_segmentation: torch.Tensor,
        img_diffusion: torch.Tensor,
        img_optical_flow: torch.Tensor,
        dynamic_func_coef,
    ) -> torch.Tensor:
        """
        Process the input images using accumulated optical flow information
        to resample the camera and diffusion images and blend them according to
        dynamic_func_coef.
        """
        # Ensure dynamic_func_coef is a list; if not, wrap it.
        if not isinstance(dynamic_func_coef, list):
            coef = [dynamic_func_coef]
        else:
            coef = dynamic_func_coef

        # Accumulate optical flow in a fixed-size buffer
        self.optical_flow_buffer.append(img_optical_flow)
        if len(self.optical_flow_buffer) > self.buffer_size:
            self.optical_flow_buffer.pop(0)

        # Compute the average optical flow over the buffer
        avg_flow = torch.stack(self.optical_flow_buffer, dim=0).mean(dim=0)  # shape (H, W, 2)

        # Prepare to warp the images using grid_sample.
        # Input images are assumed to be of shape (H, W, C)
        H, W, C = img_camera.shape
        device = img_camera.device

        # Create normalized coordinate grid in the range [-1, 1]
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, H, device=device), torch.linspace(-1, 1, W, device=device), indexing="ij")
        base_grid = torch.stack((xx, yy), dim=-1)  # (H, W, 2)

        # Normalize the averaged optical flow to be in the same scale as the grid.
        # (W-1)/2 and (H-1)/2 are the scaling factors for x and y directions.
        flow_x = avg_flow[..., 0] / ((W - 1) / 2)
        flow_y = avg_flow[..., 1] / ((H - 1) / 2)
        displacement = torch.stack((flow_x, flow_y), dim=-1)
        sampling_grid = base_grid + displacement  # (H, W, 2)

        # Define a helper function to warp an image using the computed grid
        def warp(img):
            # Convert to shape (1, C, H, W)
            img_t = img.permute(2, 0, 1).unsqueeze(0)
            warped = F.grid_sample(img_t, sampling_grid.unsqueeze(0), mode="bilinear", align_corners=True)
            # Return to (H, W, C)
            return warped.squeeze(0).permute(1, 2, 0)

        # Warp the camera and diffusion images
        disp_camera = warp(img_camera)
        disp_diff = warp(img_diffusion)

        # Blend the two warped images using the dynamic coefficient
        # coef[0] is used to control the weight of the camera image
        blended = coef[0] * disp_camera + (1 - coef[0]) * disp_diff

        # Ensure valid output range
        blended = torch.clamp(blended, 0, 255)
        return blended
