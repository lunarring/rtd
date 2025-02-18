from rtd.dynamic_processor.base_dynamic_module import BaseDynamicClass
import torch
import math

class DynamicClass(BaseDynamicClass):
    def process(self, img_camera, img_mask_segmentation, img_diffusion, img_optical_flow, dynamic_coef):
        # Compute magnitude of optical flow using the first two channels.
        flow_magnitude = torch.sqrt((img_optical_flow[..., 0] ** 2) + (img_optical_flow[..., 1] ** 2))
        
        # Expand magnitude to match camera image channels.
        mag_expanded = flow_magnitude.unsqueeze(-1)
        
        # Multiply element-wise with the camera image.
        result = img_camera * mag_expanded
        
        # Clamp output to valid image range [0, 255] and return as float32.
        return torch.clamp(result, 0, 255).to(torch.float32)
