from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray


class BaseDynamicClass(ABC):
    def __init__(self):
        """Initialize the base dynamic class."""
        pass

    @abstractmethod
    def process(
        self,
        img_camera: NDArray[np.float32],
        img_mask_segmentation: NDArray[np.float32],
        img_diffusion: NDArray[np.float32],
        dynamic_func_coef: float,
    ) -> NDArray[np.float32]:
        """
        Process the input images with the given dynamic function coefficient.

        Args:
            img_camera (np.ndarray): Input camera image. Range is [0, 255].
            img_mask_segmentation (np.ndarray): Human segmentation mask. Range is [0, 1].
            img_diffusion (np.ndarray): Diffusion image. Range is [0, 255].
            dynamic_func_coef (float): Dynamic function coefficient. Range is [0, 1].

        Returns:
            np.ndarray: Processed image as float32 numpy array
        """
        pass
