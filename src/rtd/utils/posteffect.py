import torch
import numpy as np
import torch.nn.functional as F  # retained for grid_sample and conv2d operations
import lunar_tools as lt
from PIL import Image  # added for loading particle image
import os
import random
from rtd.utils.misc_utils import get_repo_path

class Posteffect():
    def __init__(self, device='cuda:0', motion_adaptive_blend=True, use_diffusion_field=True, 
                 enable_recent_motion_color_boost=False, enable_particle_effect=True, 
                 particle_image_path=get_repo_path('materials/images', __file__), enable_upward_offset=True) -> None:
        self.device = device
        self.accumulated_frame = None  # added state to accumulate frames
        self.motion_adaptive_blend = motion_adaptive_blend
        self.use_diffusion_field = use_diffusion_field
        self.enable_recent_motion_color_boost = enable_recent_motion_color_boost
        self.enable_particle_effect = enable_particle_effect  # flag for particle effect
        self.particle_image_path = particle_image_path
        self.enable_upward_offset = enable_upward_offset  # flag for applying upward motion offset

        if self.enable_particle_effect:
            self.load_next_particle_image()

        # New attributes for smooth diffusion field interpolation
        self.accumulated_warped_particle = None
        self.diffusion_field_prev = None
        self.diffusion_field_next = None
        self.diffusion_count = 0  # counts the number of calls for interpolation
        # New attribute for tracking recent motion over the last 30 frames
        self.motion_history = None

    def load_next_particle_image(self):
        """
        Loads a random image from the particle_image_path directory and stores it as a tensor.
        Assumes particle_image_path is a directory containing image files.
        """
        # Get list of image files in directory
        valid_extensions = ('.jpg', '.jpeg', '.png')
        image_files = [f for f in os.listdir(self.particle_image_path) 
                      if f.lower().endswith(valid_extensions)]
        
        if not image_files:
            raise ValueError(f"No valid image files found in {self.particle_image_path}")
            
        # Select random image file
        random_image = random.choice(image_files)
        image_path = os.path.join(self.particle_image_path, random_image)
        
        # Load and convert to tensor
        particle_img_np = np.array(Image.open(image_path)).astype(np.float32)
        # set to solid color
        particle_img_np[:,:,:] = 228
        self.current_particle_tensor = torch.tensor(particle_img_np, device=self.device, dtype=torch.float32)
        
    def generate_smooth_random_field(self, height, width):
        """
        Generates a smooth random vector field using gaussian filtering.
        This implementation replaces the non-existent F.gaussian_blur by applying a
        gaussian filter via convolution with a custom-built gaussian kernel.
        """
        # Generate random vectors
        torch.manual_seed(np.random.randint(1000))
        random_field = torch.randn(height, width, 2, device=self.device)
        
        # Create a Gaussian kernel for convolution
        def gaussian_kernel(kernel_size, sigma, device):
            ax = torch.arange(kernel_size, dtype=torch.float32, device=device) - (kernel_size - 1) / 2.0
            gauss = torch.exp(-0.5 * (ax / sigma)**2)
            kernel1d = gauss / gauss.sum()
            # Create a 2D kernel by outer product
            kernel2d = torch.outer(kernel1d, kernel1d)
            # Reshape to (1, 1, kernel_size, kernel_size) for conv2d
            return kernel2d.unsqueeze(0).unsqueeze(0)
        
        kernel_size = 55
        sigma = 15.0
        kernel = gaussian_kernel(kernel_size, sigma, self.device)
        
        # Helper to apply gaussian blur via convolution on a single channel
        def apply_gaussian_blur(channel):
            # channel expected shape: (H, W)
            channel = channel.unsqueeze(0).unsqueeze(0)  # shape becomes (1, 1, H, W)
            padding = kernel_size // 2
            blurred = F.conv2d(channel, kernel, padding=padding)
            return blurred.squeeze(0).squeeze(0)
        
        # Apply gaussian smoothing to each of the two channels separately
        smoothed_channels = [
            apply_gaussian_blur(random_field[..., i])
            for i in range(2)
        ]
        smooth_field = torch.stack(smoothed_channels, dim=-1)
        
        # Normalize the field magnitude
        return smooth_field * 30.5  # Scale factor to control diffusion strength
        
    def warp_tensor(self, img, flow):
        """
        Warps a given image tensor using the provided optical flow.
        Assumes:
          - img is a torch.Tensor of shape (H, W, C)
          - flow is a torch.Tensor with shape (H, W, 2) in pixel displacements.
        """
        H, W = img.shape[:2]
        # Rearrange image to (1, C, H, W) for grid_sample
        img_batch = img.permute(2, 0, 1).unsqueeze(0)
        
        # Create a normalized coordinate grid in the range [-1, 1]
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=self.device),
            torch.linspace(-1, 1, W, device=self.device),
            indexing='ij'
        )
        grid = torch.stack((grid_x, grid_y), dim=2)  # shape (H, W, 2)
        
        # Convert flow from pixel displacement to normalized coordinates.
        flow_norm_x = flow[..., 0] * (2.0 / (W - 1))
        flow_norm_y = flow[..., 1] * (2.0 / (H - 1))
        flow_norm = torch.stack((flow_norm_x, flow_norm_y), dim=2)
        
        # Compute the new grid positions by adding the normalized flow
        grid_warp = grid + flow_norm
        grid_warp = grid_warp.unsqueeze(0)  # add batch dimension
        
        # Warp the image using grid_sample.
        warped_img = F.grid_sample(img_batch, grid_warp, mode='bilinear', padding_mode='border', align_corners=True)
        # Return to original shape (H, W, C)
        warped_img = warped_img.squeeze(0).permute(1, 2, 0)
        return warped_img

    def process(self, img_diffusion, img_mask_segmentation, img_optical_flow,
                flow_gain, reverb_gain, background_image_gain, mod_button1, sound_volume_modulation):
        # Convert inputs to torch tensors; ensure they are float32 for processing.
        torch_img_diffusion = torch.tensor(np.asarray(img_diffusion), device=self.device, dtype=torch.float32)
        torch_img_mask_segmentation = torch.tensor(np.asarray(img_mask_segmentation), device=self.device, dtype=torch.float32)
        torch_img_optical_flow = torch.tensor(np.asarray(img_optical_flow), device=self.device, dtype=torch.float32)

        torch_img_mask_segmentation = lt.resize(torch_img_mask_segmentation, size=(torch_img_diffusion.shape[0], torch_img_diffusion.shape[1]))
        torch_img_optical_flow = lt.resize(torch_img_optical_flow, size=(torch_img_diffusion.shape[0], torch_img_diffusion.shape[1]))
        
        # Apply constant upward offset to the Y component of the optical flow if enabled
        if self.enable_upward_offset:
            upward_offset = 1.0  # negative value to shift upward
            torch_img_optical_flow[..., 1] += upward_offset
        
        # Compute motion magnitude from optical flow for both motion adaptation and color boost.
        motion_magnitude = torch.sqrt(torch_img_optical_flow[..., 0]**2 + torch_img_optical_flow[..., 1]**2)
        
        # Update motion history if color boost effect is enabled.
        if self.enable_recent_motion_color_boost:
            wnd = 5.0
            threshold = 1.0
            current_motion = (motion_magnitude > threshold).float()
            if self.motion_history is None:
                self.motion_history = current_motion * wnd
            else:
                self.motion_history = torch.clamp(self.motion_history - 1, min=0)
                self.motion_history = torch.where(current_motion > 0, torch.full_like(self.motion_history, wnd), self.motion_history)
            recent_motion_mask = (self.motion_history > 0).float()
        
        # Initialize the accumulated frame if not set
        if self.accumulated_frame is None:
            self.accumulated_frame = torch_img_diffusion

        # Apply modulation to the optical flow using flow_gain.
        modulated_flow = torch_img_optical_flow * flow_gain * 5
        
        # Add smooth random diffusion field if enabled with interpolation over 20 calls
        if self.use_diffusion_field:
            H, W = torch_img_diffusion.shape[0], torch_img_diffusion.shape[1]
            # Initialize diffusion fields if not already initialized
            if self.diffusion_field_prev is None or self.diffusion_field_next is None:
                self.diffusion_field_prev = self.generate_smooth_random_field(H, W)
                self.diffusion_field_next = self.generate_smooth_random_field(H, W)
                self.diffusion_count = 0
            # Compute interpolation ratio
            ratio = self.diffusion_count / 20.0
            diffusion_field = (1 - ratio) * self.diffusion_field_prev + ratio * self.diffusion_field_next
            # Increment call counter and refresh fields every 20 calls
            self.diffusion_count += 1
            if self.diffusion_count >= 20:
                self.diffusion_field_prev = self.diffusion_field_next
                self.diffusion_field_next = self.generate_smooth_random_field(H, W)
                self.diffusion_count = 0
            modulated_flow = modulated_flow + diffusion_field * flow_gain * 5
        
        # Warp the previous accumulated frame based on the modulated optical flow.
        warped_accum = self.warp_tensor(self.accumulated_frame, modulated_flow)
        
        # Compute base blend weights: for background (mask==0) use 0.2 and for human (mask==1) use 0.7
        # This means background will retain 80% of previous frame while human areas update with 70% of new frame
        base_alpha = (1 - torch_img_mask_segmentation) * 0.2 + torch_img_mask_segmentation * 0.7
        
        if self.motion_adaptive_blend:
            # Use precomputed motion magnitude to calculate average motion.
            avg_motion = torch.mean(motion_magnitude)
            
            # Use a sigmoid-like mapping to [0,1] range with sensitivity controlled by reverb_gain
            motion_factor = (1-reverb_gain) * 3 * avg_motion.item()
            motion_factor = np.clip(motion_factor, 0, 1)
            
            # Scale base_alpha by motion_factor
            effective_alpha = base_alpha * motion_factor
        else:
            # Original behavior when motion adaptive blend is disabled
            effective_alpha = base_alpha * (1 - reverb_gain)
        
        # Compute the new accumulated frame with exponential moving average blend.
        output_to_render = effective_alpha * torch_img_diffusion + (1 - effective_alpha) * warped_accum
        
        # Apply color boost in areas with recent motion if the effect is enabled.
        if self.enable_recent_motion_color_boost:
            # Helper function to boost colors: increases brightness and saturation.
            def apply_color_boost(img, mask, brightness_factor=1.05, saturation_factor=1.05):
                # img: tensor of shape (H, W, C), mask: tensor of shape (H, W) with values in [0,1]
                grey = img.mean(dim=2, keepdim=True)
                saturated = grey + saturation_factor * (img - grey)
                boosted = saturated * brightness_factor
                mask_expanded = mask.unsqueeze(2)  # expand mask to (H, W, 1)
                return mask_expanded * boosted + (1 - mask_expanded) * img

            output_to_render = apply_color_boost(output_to_render, recent_motion_mask)
        
        # New particle effect implementation: load particle image from file, displace it with optical flow,
        # and add the resulting resampled image to the accumulated frame.
        if self.enable_particle_effect and self.particle_image_path is not None and mod_button1:

            if np.random.rand() < 0.002:
                self.load_next_particle_image()

            # Load the particle image from the specified file path
            particle_img = self.current_particle_tensor
            # Resize the particle image to match the dimensions of the diffusion image
            particle_img = lt.resize(particle_img[:,:,:3], size=(torch_img_diffusion.shape[0], torch_img_diffusion.shape[1]))
            # Compute the flow for the particles using optical flow modulated by flow_gain
            particle_flow = torch_img_optical_flow * flow_gain
            if self.accumulated_warped_particle is None:
                # Resample/displace the particle image using the optical flow
                warped_particle = self.warp_tensor(particle_img, particle_flow)
                self.accumulated_warped_particle = warped_particle
            else:
                particle_img = self.accumulated_warped_particle * 0.9 + particle_img * 0.01
                warped_particle = self.warp_tensor(particle_img, particle_flow)

                # Reconstruct the final output image as it will be used:
                # Apply the same limiter logic to compute a temporary warped_particle

                # Compute a grayscale (luminance) image using standard perceptual weights.
                grayscale = (0.299 * output_to_render[..., 0] +
                             0.587 * output_to_render[..., 1] +
                             0.114 * output_to_render[..., 2])
                # Estimate overall brightness as the mean intensity.
                brightness = torch.mean(grayscale)
                # Determine normalization factor: assume pixel range of [0,255] if max > 1, otherwise [0,1].
                norm_factor = 255.0 if torch.max(output_to_render) > 1.0 else 1.0
                overall_brightness = torch.clamp(brightness / norm_factor, 0.0, 1.0).item()

                brightness_limiter = 1 / (1.0 + overall_brightness*0.5)

                #  limiter = reverb_gain ** 2
                limiter = background_image_gain
                warped_particle *= limiter * brightness_limiter

            self.accumulated_warped_particle = warped_particle

            # Add the warped particle image to the accumulated frame
            output_for_diffusion = output_to_render + warped_particle
        else:
            output_for_diffusion = output_to_render
        
        # Update the stored accumulated frame (detached to avoid any gradient backpropagation)
        self.accumulated_frame = output_for_diffusion.detach()

        output_to_render[:,:,0] *= (1+sound_volume_modulation*0.5)
        
        # Return the processed image which has both smooth accumulated effects and fluid flow.
        return output_to_render.cpu().numpy(), output_for_diffusion.cpu().numpy()