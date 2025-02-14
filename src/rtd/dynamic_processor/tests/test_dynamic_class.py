import unittest
import numpy as np
import time
from dynamic_module import DynamicClass

class DynamicClassTests(unittest.TestCase):
    def setUp(self):
        # Create sample test data with known shapes and float32 data type
        self.shape = (1024, 1024, 3)
        self.img_camera = np.ones(self.shape, dtype=np.float32)
        self.img_mask = np.zeros(self.shape, dtype=np.float32)
        self.img_diffusion = np.full(self.shape, 0.5, dtype=np.float32)
        self.dynamic_coef = 0.5
        self.processor = DynamicClass()

    def test_output_shape(self):
        """Test that output shape matches input shape"""
        result = self.processor.process(self.img_camera, self.img_mask, self.img_diffusion, self.dynamic_coef)
        self.assertEqual(result.shape, self.img_camera.shape)

    def test_data_type(self):
        """Test that output data type is float32"""
        result = self.processor.process(self.img_camera, self.img_mask, self.img_diffusion, self.dynamic_coef)
        self.assertEqual(result.dtype, np.float32)

    def test_performance(self):
        """Test that processing time is under 100ms"""
        start_time = time.time()
        self.processor.process(self.img_camera, self.img_mask, self.img_diffusion, self.dynamic_coef)
        elapsed_time = (time.time() - start_time) * 1000  # ms
        self.assertLess(elapsed_time, 100, f"Processing took {elapsed_time:.2f}ms (>100ms)")

if __name__ == "__main__":
    unittest.main()
