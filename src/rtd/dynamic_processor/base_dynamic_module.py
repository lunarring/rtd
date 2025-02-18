from abc import ABC, abstractmethod
import torch
import numpy as np
from numpy.typing import NDArray


class BaseDynamicClass(ABC):
    def __init__(self):
        """Initialize the base dynamic class."""
        pass

    @abstractmethod
    def process(
        self,
        img_camera: torch.Tensor,
        img_mask_segmentation: torch.Tensor,
        img_diffusion: torch.Tensor,
        img_optical_flow: torch.Tensor,
        dynamic_func_coef: list,
    ) -> torch.Tensor:
        """
        Process the input images with the given dynamic function coefficient.

        Args:
            img_camera (torch.Tensor), range [0, 255]. Tensor has HEIGHT x WIDTH x 3. This is the webcam camera image in RGB format, showing people in front of the camera.
            img_mask_segmentation (torch.Tensor): range [0, 1]. Tensor has HEIGHT x WIDTH. This is the human segmentation mask, where 1 means a human and 0 means background, for applying the mask we need to multiply an image with it.
            img_diffusion (torch.Tensor): Diffusion image. Range is [0, 255]. Tensor has HEIGHT x WIDTH x 3. This is the last image that the AI process yielded in the previous iteration.
            img_optical_flow (torch.Tensor): Range indicates the optical flow of the camera image. Tensor has HEIGHT x WIDTH x 2. The channels contain the x and y components of the motion vector at each pixel. Thus the values are typically very low, usually between -20 and 20.
            dynamic_func_coef (list): List of floats, the size of the list you need to find out. Range is [0, 1]. Each item of the list contains a dynamic coefficient, which is a float parameter that is given by the user and we want to map it to modulate something interesting.

        Returns:
            torch.Tensor: Processed image as float32 torch tensor
        """
        pass
