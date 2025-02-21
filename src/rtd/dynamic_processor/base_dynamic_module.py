from abc import ABC, abstractmethod
import torch
import numpy as np
from numpy.typing import NDArray


class BaseDynamicClass(ABC):
    def __init__(self):
        """Initialize the base dynamic class."""
        pass

    def assert_inputs(
        self,
        img_camera: torch.Tensor,
        img_mask_segmentation: torch.Tensor,
        img_diffusion: torch.Tensor,
        img_optical_flow: torch.Tensor,
        dynamic_coef: float,
    ) -> None:
        """
        Validate input shapes and types.

        Args:
            Same as process() method.

        Raises:
            AssertionError: If any of the input validations fail.
        """
        # Check that camera, mask and diffusion have same shape
        msg = "Camera shape {} does not match mask shape {}"
        assert img_camera.shape == img_mask_segmentation.shape[:2] + (3,), msg.format(img_camera.shape, img_mask_segmentation.shape)

        msg = "Camera shape {} does not match diffusion shape {}"
        assert img_camera.shape == img_diffusion.shape, msg.format(img_camera.shape, img_diffusion.shape)

        # Check optical flow height/width matches camera
        msg = "Optical flow height {} does not match camera height {}"
        assert img_optical_flow.shape[0] == img_camera.shape[0], msg.format(img_optical_flow.shape[0], img_camera.shape[0])

        msg = "Optical flow width {} does not match camera width {}"
        assert img_optical_flow.shape[1] == img_camera.shape[1], msg.format(img_optical_flow.shape[1], img_camera.shape[1])

        # Verify dynamic_coef is a scalar float
        msg = "dynamic_coef must be a float, got {}"
        assert isinstance(dynamic_coef, float), msg.format(type(dynamic_coef))

        msg = "dynamic_coef must be a scalar, not a sequence"
        assert not isinstance(dynamic_coef, (list, tuple, np.ndarray)), msg

    @abstractmethod
    def process(
        self,
        img_camera: torch.Tensor,
        img_mask_segmentation: torch.Tensor,
        img_diffusion: torch.Tensor,
        img_optical_flow: torch.Tensor,
        dynamic_coef: float,
    ) -> torch.Tensor:
        """
        Process the input images with the given dynamic function coefficient.

        Args:
            img_camera (torch.Tensor), range [0, 255]. Tensor has HEIGHT x WIDTH x 3. This is the webcam camera image in RGB format, showing people in front of the camera.
            img_mask_segmentation (torch.Tensor): range [0, 1]. Tensor has HEIGHT x WIDTH x 1. This is the human segmentation mask, where 1 means a human and 0 means background, for applying the mask we need to multiply an image with it.
            img_diffusion (torch.Tensor): Diffusion image. Range is [0, 255]. Tensor has HEIGHT x WIDTH x 3. This is the last image that was the result of the last diffusion process, which had the output of this function as input, thus it is recursive.
            img_optical_flow (torch.Tensor): Range indicates the optical flow of the camera image. Tensor has HEIGHT x WIDTH x 2. The channels contain the x and y components of the motion vector at each pixel. Thus the values are typically very low, usually between -20 and 20.
            dynamic_coef (float): Range is [0, 1]. Float parameter is given by the user and we want to map each of them to modulate something interesting.

        Returns:
            torch.Tensor: Processed image as float32 torch tensor
        """
        pass
