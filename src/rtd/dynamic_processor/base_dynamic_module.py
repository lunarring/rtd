from abc import ABC, abstractmethod
import numpy as np


class BaseDynamicClass(ABC):
    def __init__(self):
        """Initialize the base dynamic class."""
        pass

    @abstractmethod
    def process(
        self,
        img_camera: np.ndarray,
        img_mask_segmentation: np.ndarray,
        img_diffusion: np.ndarray,
        dynamic_func_coef: float,
    ) -> np.ndarray:
        """
        Process the input images with the given dynamic function coefficient.

        Args:
            img_camera (np.ndarray): Input camera image as float32 numpy array
            img_mask_segmentation (np.ndarray): Human segmentation mask as float32
                numpy array
            img_diffusion (np.ndarray): Diffusion image as float32 numpy array
            dynamic_func_coef (float): Dynamic function coefficient

        Returns:
            np.ndarray: Processed image as float32 numpy array
        """
        pass
