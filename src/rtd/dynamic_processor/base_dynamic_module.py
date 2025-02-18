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
        dynamic_func_coef: float,
    ) -> torch.Tensor:
        """
        Process the input images with the given dynamic function coefficient.

        Args:
            img_camera (torch.Tensor): Input camera image. Range is [0, 255].
            img_mask_segmentation (torch.Tensor): Human segmentation mask. Range is [0, 1].
            img_diffusion (torch.Tensor): Diffusion image. Range is [0, 255].
            dynamic_func_coef (float): Dynamic function coefficient. Range is [0, 1].

        Returns:
            torch.Tensor: Processed image as float32 torch tensor
        """
        pass
