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
        img_camera: NDArray[np.uint8],
        img_mask_segmentation: NDArray[np.uint8],
        img_diffusion: NDArray[np.uint8],
        dynamic_func_coef: float,
    ) -> NDArray[np.float32]:
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
