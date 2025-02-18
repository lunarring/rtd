import numpy as np
import cv2
import torch
import torch.nn.functional as F
from rtd.utils.FastFlowNet_v2 import FastFlowNet
from lunar_tools.cam import WebCam
import time
import argparse
from matplotlib import pyplot as plt

class OpticalFlowEstimator:
    """
    A class to estimate optical flow between consecutive frames using the FastFlowNet model.

    Attributes:
        device (torch.device): The device to run the model on (e.g., 'cuda:0').
        model (FastFlowNet): The FastFlowNet model for optical flow estimation.
        div_flow (float): A scaling factor for the flow output.
        div_size (int): The size to which input images are padded for processing.
        prev_img (np.ndarray): The previous image frame for flow calculation.
    """
    def __init__(self, model_path='./checkpoints/fastflownet_ft_mix.pth', div_flow=20.0, div_size=64, return_numpy=True, device='cuda:0'):
        """
        Initializes the OpticalFlowEstimator with the specified model path, flow division factor, 
        division size, and device.

        Args:
            model_path (str): Path to the pre-trained model weights.
            div_flow (float): Scaling factor for the flow output.
            div_size (int): Size to which input images are padded for processing.
            device (str): Device to run the model on (e.g., 'cuda:0').
        """
        self.device = torch.device(device)
        self.model = FastFlowNet().to(self.device).eval()
        self.model.load_state_dict(torch.load(model_path))
        self.div_flow = div_flow
        self.div_size = div_size
        self.prev_img = None
        self.return_numpy = return_numpy

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

    def get_optflow(self, img, low_pass_kernel_size=0):
        """
        Computes the optical flow between the current and previous image frames.

        Args:
            img (np.ndarray): The current image frame.
            low_pass_kernel_size (int): The kernel size for the optional low-pass filter.

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
        input_size = (
            int(self.div_size * np.ceil(height / self.div_size)), 
            int(self.div_size * np.ceil(width / self.div_size))
        )
        
        if input_size != (height, width):
            img1 = F.interpolate(img1, size=input_size, mode='bilinear', align_corners=False)
            img2 = F.interpolate(img2, size=input_size, mode='bilinear', align_corners=False)

        # Prepare input tensor and run model
        input_t = torch.cat([img1, img2], 1)
        output = self.model(input_t).data

        # Process flow output
        flow = self.div_flow * F.interpolate(output, size=input_size, mode='bilinear', align_corners=False)
        
        if input_size != (height, width):
            scale_h = height / input_size[0]
            scale_w = width / input_size[1]
            flow = F.interpolate(flow, size=(height, width), mode='bilinear', align_corners=False)
            flow[:, 0, :, :] *= scale_w
            flow[:, 1, :, :] *= scale_h

        # Apply low-pass filter if specified
        flow = self.low_pass_filter(flow, low_pass_kernel_size)

        flow = flow[0].permute(1, 2, 0)
        if self.return_numpy:
            flow = flow.cpu().numpy()

        self.prev_img = img
        return flow

if __name__ == "__main__":
    show_histogram = True  # Simple flag to control histogram display
    flow_range = 20  # Increased range for visualization (-20 to +20)
    
    cam = WebCam()
    opt_flow_estimator = OpticalFlowEstimator()
    
    if show_histogram:
        plt.ion()
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
        plt.tight_layout(pad=3.0)

    while True:
        start_time = time.time()

        img = np.flip(cam.get_img(), axis=1).copy()
        flow = opt_flow_estimator.get_optflow(img, low_pass_kernel_size=55)

        if flow is not None:
            # Normalize and display raw flow components
            flow_x = flow[..., 0]
            flow_y = flow[..., 1]
            
            # Scale flows for visualization (-20 to 20 range to 0-255)
            flow_x_vis = np.clip((flow_x + flow_range) * (255/(2*flow_range)), 0, 255).astype(np.uint8)
            flow_y_vis = np.clip((flow_y + flow_range) * (255/(2*flow_range)), 0, 255).astype(np.uint8)
            
            # Stack horizontally for display
            combined_flow = np.hstack((flow_x_vis, flow_y_vis))
            cv2.imshow('Raw Flow (X | Y)', combined_flow)
            
            if show_histogram:
                # Clear previous plots
                ax1.clear()
                ax2.clear()
                ax3.clear()
                
                # Plot X flow histogram
                ax1.hist(flow_x.flatten(), bins=50, range=(-flow_range, flow_range))
                ax1.set_title('X Flow Distribution')
                ax1.set_xlabel('X Magnitude')
                ax1.set_ylabel('Frequency')
                
                # Plot Y flow histogram
                ax2.hist(flow_y.flatten(), bins=50, range=(-flow_range, flow_range))
                ax2.set_title('Y Flow Distribution')
                ax2.set_xlabel('Y Magnitude')
                
                # Plot combined magnitude histogram
                magnitude = np.sqrt(flow_x**2 + flow_y**2)
                ax3.hist(magnitude.flatten(), bins=50, range=(0, flow_range))
                ax3.set_title('Flow Magnitude')
                ax3.set_xlabel('Magnitude')
                
                plt.draw()
                plt.pause(0.001)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        end_time = time.time()
        print(f"Iteration time: {end_time - start_time:.4f} seconds")

    cv2.destroyAllWindows()
    if show_histogram:
        plt.close()
