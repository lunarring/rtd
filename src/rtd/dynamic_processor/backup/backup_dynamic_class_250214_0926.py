from rtd.dynamic_processor.base_dynamic_module import BaseDynamicClass
import numpy as np

class DynamicClass(BaseDynamicClass):
    def process(
        self,
        img_camera: np.ndarray,
        img_mask_segmentation: np.ndarray,
        img_diffusion: np.ndarray,
        dynamic_func_coef: float,
    ) -> np.ndarray:
        # Fill the human segmentation mask with random noise.
        noise = np.random.rand(*img_mask_segmentation.shape).astype(np.float32)
        return noise
