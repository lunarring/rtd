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
        img_diffusion: torch.Tensor,
        img_mask_segmentation: torch.Tensor,
        img_optical_flow: torch.Tensor,
        dynamic_coef: float,
    ) -> torch.Tensor:
        """
        Process the input images with the given dynamic function coefficient.

        Args:
            img_diffusion (torch.Tensor): Diffusion image. Range is [0, 255]. Tensor has HEIGHT x WIDTH x 3. This is the last image that was the result of the last diffusion process, which had the output of this function as input, thus it is recursive.
            img_mask_segmentation (torch.Tensor): range [0, 1]. Tensor has HEIGHT x WIDTH x 1. This is the human segmentation mask, where 1 means a human and 0 means background, for applying the mask we need to multiply an image with it.
            img_optical_flow (torch.Tensor): Range indicates the optical flow of the camera image. Tensor has HEIGHT x WIDTH x 2. The channels contain the x and y components of the motion vector at each pixel. Thus the values are typically very low, usually between -20 and 20.
            dynamic_coef (float): Range is [0, 1]. Float parameter is given by the user and we want to map each of them to modulate something interesting.

        Returns:
            torch.Tensor: Processed image as float32 torch tensor
        """
        pass
