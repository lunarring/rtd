import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
from .FastFlowNet_v2 import FastFlowNet
from lunar_tools.cam import WebCam
import time
import argparse
from matplotlib import pyplot as plt
import lunar_tools as lt
from rtd.utils.misc_utils import get_repo_path

class OpticalFlowEstimator:
    """
    A class to estimate optical flow between consecutive frames using the FastFlowNet model.

    Attributes:
        device (torch.device): The device to run the model on (e.g., 'cuda:0').
        model (FastFlowNet): The FastFlowNet model for optical flow estimation.
        div_flow (float): A scaling factor for the flow output.
        div_size (int): The size to which input images are padded for processing.
        prev_img (np.ndarray): The previous image frame for flow calculation.
        flow_history (list): List of previous flow fields for temporal smoothing.
        use_ema (bool): Whether to use exponential moving average for temporal smoothing.
    """

    def __init__(self, model_path=None, div_flow=20.0, div_size=64, 
                 return_numpy=True, device="cuda:0", use_ema=False):
        """
        Initializes the OpticalFlowEstimator with the specified model path, flow division factor,
        division size, and device.

        Args:
            model_path (str): Path to the pre-trained model weights. If None, uses default path.
            div_flow (float): Scaling factor for the flow output.
            div_size (int): Size to which input images are padded for processing.
            device (str): Device to run the model on (e.g., 'cuda:0').
            use_ema (bool): If True, use exponential moving average for temporal smoothing.
                          If False, use simple averaging (default).
        """
        self.device = torch.device(device)
        self.model = FastFlowNet().to(self.device).eval()
        
        # Use default path if none provided
        if model_path is None:
            model_path = get_repo_path("checkpoints/fastflownet_ft_mix.pth", __file__)
        
        self.model.load_state_dict(torch.load(model_path))
        self.div_flow = div_flow
        self.div_size = div_size
        self.prev_img = None
        self.return_numpy = return_numpy
        self.flow_history = []
        self.use_ema = use_ema

    def centralize(self, img1, img2):
        """
        Centralizes the input images by subtracting the mean RGB value.

        Args:
            img1 (torch.Tensor): The first image tensor.
            img2 (torch.Tensor): The second image tensor.

        Returns:
            tuple: Centralized images and the mean RGB value.
        """
        b, c, h, w = img1.shape
        rgb_mean = torch.cat([img1, img2], dim=2).view(b, c, -1).mean(2).view(b, c, 1, 1)
        return img1 - rgb_mean, img2 - rgb_mean, rgb_mean

    def low_pass_filter(self, flow, kernel_size):
        """
        Applies a low-pass filter to the flow field to smooth it.

        Args:
            flow (torch.Tensor): The flow field tensor.
            kernel_size (int): The size of the kernel for the low-pass filter.

        Returns:
            torch.Tensor: The smoothed flow field.
        """
        if kernel_size > 0:
            padding = kernel_size // 2
            kernel = torch.ones((2, 1, kernel_size, kernel_size), device=self.device) / (kernel_size * kernel_size)
            flow = F.conv2d(flow, kernel, padding=padding, groups=2)
        return flow

    def get_optflow(self, img, low_pass_kernel_size=0, window_length=1):
        """
        Computes the optical flow between the current and previous image frames.

        Args:
            img (np.ndarray): The current image frame.
            low_pass_kernel_size (int): The kernel size for the optional low-pass filter.
            window_length (int): Number of frames to use for temporal smoothing.

        Returns:
            np.ndarray: The computed optical flow, or None if there is no previous image.
        """
        if self.prev_img is None:
            self.prev_img = img
            return None

        # Convert images to tensors and centralize
        img1 = torch.from_numpy(self.prev_img).float().permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
        img2 = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
        img1, img2, _ = self.centralize(img1, img2)

        # Calculate input size and interpolate if necessary
        height, width = img1.shape[-2:]
        input_size = (int(self.div_size * np.ceil(height / self.div_size)), int(self.div_size * np.ceil(width / self.div_size)))

        if input_size != (height, width):
            img1 = F.interpolate(img1, size=input_size, mode="bilinear", align_corners=False)
            img2 = F.interpolate(img2, size=input_size, mode="bilinear", align_corners=False)

        # Prepare input tensor and run model
        input_t = torch.cat([img1, img2], 1)
        output = self.model(input_t).data

        # Process flow output
        flow = self.div_flow * F.interpolate(output, size=input_size, mode="bilinear", align_corners=False)

        if input_size != (height, width):
            scale_h = height / input_size[0]
            scale_w = width / input_size[1]
            flow = F.interpolate(flow, size=(height, width), mode="bilinear", align_corners=False)
            flow[:, 0, :, :] *= scale_w
            flow[:, 1, :, :] *= scale_h

        # Apply low-pass filter if specified
        flow = self.low_pass_filter(flow, low_pass_kernel_size)

        # Apply temporal smoothing if window_length > 1
        self.flow_history.append(flow)
        if len(self.flow_history) > window_length:
            self.flow_history.pop(0)
        if len(self.flow_history) > 1:
            if self.use_ema:
                # Apply exponential moving average with a decay factor
                alpha = 0.2  # Decay factor (adjust as needed)
                smoothed_flow = self.flow_history[0]
                for f in self.flow_history[1:]:
                    smoothed_flow = alpha * f + (1 - alpha) * smoothed_flow
                flow = smoothed_flow
            else:
                # Simple averaging
                flow = torch.stack(self.flow_history).mean(dim=0)

        flow = flow[0].permute(1, 2, 0)
        if self.return_numpy:
            flow = flow.cpu().numpy()

        self.prev_img = img
        return flow


if __name__ == "__main__":
    import lunar_tools as lt

    show_histogram = False  # Simple flag to control histogram display
    flow_range = 20  # Increased range for visualization (-20 to +20)

    shape_cam = (576, 1024)
    cam = WebCam(shape_hw=shape_cam)
    renderer = lt.Renderer(width=shape_cam[1], height=shape_cam[0], backend="pygame")
    opt_flow_estimator = OpticalFlowEstimator()

    if show_histogram:
        plt.ion()
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
        plt.tight_layout(pad=3.0)

    while True:
        start_time = time.time()

        cam_img = cam.get_img()
        cam_img = np.flip(cam_img, axis=1).astype(np.float32)
        flow = opt_flow_estimator.get_optflow(cam_img.copy(), low_pass_kernel_size=35, window_length=45)

        if flow is not None:
            # Extract x and y components of flow
            flow_x = flow[..., 0]
            flow_y = flow[..., 1]

            # Create empty RGB array for visualization
            flow_rgb = np.zeros((*flow.shape[:-1], 3), dtype=np.float32)

            # Calculate magnitude and normalize to 0-255 range
            magnitude = np.sqrt(flow_x**2 + flow_y**2)
            max_magnitude = np.max(magnitude)
            normalized_magnitude = magnitude * 50
            normalized_magnitude = np.clip(normalized_magnitude, 0, 255)

            # Map flow to RGB channels:
            # Positive x flow -> red
            # Negative x flow -> blue
            flow_rgb[..., 0] = np.where(flow_x > 0, normalized_magnitude * (flow_x / max_magnitude), 0)
            flow_rgb[..., 2] = np.where(flow_x < 0, normalized_magnitude * (-flow_x / max_magnitude), 0)

            # Positive y flow -> green
            # Negative y flow -> blue + green mix
            flow_rgb[..., 1] = np.where(
                flow_y > 0,
                normalized_magnitude * (flow_y / max_magnitude),
                np.where(flow_y < 0, normalized_magnitude * (-flow_y / max_magnitude) * 0.5, 0),
            )
            flow_rgb[..., 2] += np.where(flow_y < 0, normalized_magnitude * (-flow_y / max_magnitude) * 0.5, 0)

            renderer.render(flow_rgb)

            if show_histogram:
                # Clear previous plots
                ax1.clear()
                ax2.clear()
                ax3.clear()

                # Plot X flow histogram
                ax1.hist(flow_x.flatten(), bins=50, range=(-flow_range, flow_range))
                ax1.set_title("X Flow Distribution")
                ax1.set_xlabel("X Magnitude")
                ax1.set_ylabel("Frequency")

                # Plot Y flow histogram
                ax2.hist(flow_y.flatten(), bins=50, range=(-flow_range, flow_range))
                ax2.set_title("Y Flow Distribution")
                ax2.set_xlabel("Y Magnitude")

                # Plot magnitude histogram
                ax3.hist(magnitude.flatten(), bins=50, range=(0, flow_range))
                ax3.set_title("Flow Magnitude")
                ax3.set_xlabel("Magnitude")

                plt.draw()
                plt.pause(0.001)

        end_time = time.time()
        print(f"Iteration time: {end_time - start_time:.4f} seconds")

    if show_histogram:
        plt.close()
