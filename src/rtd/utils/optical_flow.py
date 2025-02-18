import numpy as np
import cv2
import torch
import torch.nn.functional as F
from rtd.utils.FastFlowNet_v2 import FastFlowNet
from lunar_tools.cam import WebCam
import time

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
    def __init__(self, model_path='./checkpoints/fastflownet_ft_mix.pth', div_flow=20.0, div_size=64, device='cuda:0'):
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
        flow = flow[0].cpu().permute(1, 2, 0).numpy()
        
        self.prev_img = img
        return flow

if __name__ == "__main__":
    cam = WebCam(cam_id=-1)
    opt_flow_estimator = OpticalFlowEstimator()

    while True:
        start_time = time.time()  # Start timing

        img = np.flip(cam.get_img(), axis=1).copy()
        flow = opt_flow_estimator.get_optflow(img, low_pass_kernel_size=55)

        if flow is not None:
            height, width = flow.shape[:2]

            # Convert the optical flow to a 3-channel image for display
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv = np.zeros((height, width, 3), dtype=np.uint8)
            hsv[..., 0] = angle * 180 / np.pi / 2  # Hue
            hsv[..., 1] = 255  # Saturation
            hsv[..., 2] = np.clip(magnitude * 5, 0, 255)
            flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            # Resize the image to twice its original size for display
            flow_bgr_resized = cv2.resize(flow_bgr, (width * 2, height * 2), interpolation=cv2.INTER_LINEAR)
            cv2.imshow('Optical Flow', flow_bgr_resized)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        end_time = time.time()  # End timing
        print(f"Iteration time: {end_time - start_time:.4f} seconds")  # Print iteration time

    cv2.destroyAllWindows()
