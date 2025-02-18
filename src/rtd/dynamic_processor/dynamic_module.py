import torch
import torch.nn.functional as F
from rtd.dynamic_processor.base_dynamic_module import BaseDynamicClass

class DynamicClass(BaseDynamicClass):
    def __init__(self):
        self._grid_cache = {}

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
        img_camera = img_camera * (1 - img_mask_segmentation)
        
        # Handle case when optical flow is omitted (4-argument call)
        if isinstance(img_optical_flow, float):
            dynamic_func_coef = img_optical_flow
            img_optical_flow = None

        if img_optical_flow is None:
            # Liquid blend: reduce diffusion image influence, making it more like a liquid.
            liquid_coef = dynamic_func_coef * 0.5
            blended = img_camera * (1 - liquid_coef) + img_diffusion * liquid_coef
        else:
            # Use optical flow to create a wave-like distortion on the diffusion image.
            # Only use the first two channels of optical flow.
            flow = img_optical_flow[..., :2]
            H, W, _ = img_diffusion.shape
            key = (H, W, img_diffusion.device)
            if key in self._grid_cache:
                grid = self._grid_cache[key]
            else:
                grid_y, grid_x = torch.meshgrid(
                    torch.linspace(-1, 1, H, device=img_diffusion.device),
                    torch.linspace(-1, 1, W, device=img_diffusion.device),
                    indexing='ij'
                )
                grid = torch.stack((grid_x, grid_y), dim=2)
                self._grid_cache[key] = grid
            # Normalize optical flow displacement relative to image dimensions.
            disp_x = (flow[..., 0] / (W / 2)) * dynamic_func_coef
            disp_y = (flow[..., 1] / (H / 2)) * dynamic_func_coef
            disp = torch.stack((disp_x, disp_y), dim=2)
            warped_grid = grid + disp
            # Prepare diffusion image for grid_sample: (N, C, H, W)
            diffusion_batch = img_diffusion.permute(2, 0, 1).unsqueeze(0)
            warped_grid = warped_grid.unsqueeze(0)
            warped_diffusion = F.grid_sample(diffusion_batch, warped_grid, mode='bilinear', padding_mode='border', align_corners=True)
            warped_diffusion = warped_diffusion.squeeze(0).permute(1, 2, 0)
            liquid_coef = dynamic_func_coef * 0.5
            # Blend camera image with warped diffusion image using reduced diffusion weight.
            blended = img_camera * (1 - liquid_coef) + warped_diffusion * liquid_coef

        return torch.clamp(blended, 0, 255)
