import os
import unittest
import numpy as np
import torch
from src.rtd.dynamic_processor.processor_dynamic_module import DynamicProcessor

class TestDynamicProcessor(unittest.TestCase):
    def test_tensor_conversion_and_output(self):
        original_tensor = torch.tensor
        calls = []

        def fake_tensor(data, *args, **kwargs):
            calls.append(kwargs.get('device', None))
            return original_tensor(data, *args, **kwargs)

        torch.tensor = fake_tensor
        try:
            processor = DynamicProcessor()
            # Ensure the dynamic module file exists so process doesn't exit early.
            with open(processor.fp_func, "w") as f:
                f.write("class DynamicClass:\n"
                        "    def process(self, img_camera, img_mask_segmentation, img_diffusion, dynamic_func_coef=0.5):\n"
                        "        return img_camera\n")
            dummy_image = np.zeros((256, 256, 3), dtype=np.uint8)
            output = processor.process(dummy_image, dummy_image, dummy_image, dynamic_func_coef=0.5)

            self.assertIn('cuda', calls, "torch.tensor was not called with device='cuda'")
            self.assertTrue(isinstance(output, np.ndarray), "Output is not a numpy array")
            self.assertEqual(output.shape, dummy_image.shape, "Output shape does not match input shape")
        finally:
            torch.tensor = original_tensor
            if os.path.exists(processor.fp_func):
                os.remove(processor.fp_func)

if __name__ == "__main__":
    unittest.main()
