import unittest
import numpy as np
import time
import os
import sys

# Add parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dynamic_module import process


class TestDynamicModule(unittest.TestCase):
    def setUp(self):
        # Create sample test data with known shapes and float32 data type
        self.shape = (1024, 1024, 3)  # Example shape for RGB image
        self.img_camera = np.ones(self.shape, dtype=np.float32)
        self.img_mask = np.zeros(self.shape, dtype=np.float32)
        self.img_diffusion = np.full(self.shape, 0.5, dtype=np.float32)

    def test_output_shape(self):
        """Test that output shape matches input shape"""
        result = process(self.img_camera, self.img_mask, self.img_diffusion)
        self.assertEqual(result.shape, self.img_camera.shape)

    def test_data_type(self):
        """Test that output data type is float32"""
        result = process(self.img_camera, self.img_mask, self.img_diffusion)
        self.assertEqual(result.dtype, np.float32)

    def test_performance(self):
        """Test that processing time is under 100ms"""
        start_time = time.time()
        process(self.img_camera, self.img_mask, self.img_diffusion)
        elapsed_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        self.assertLess(elapsed_time, 100, f"Processing took {elapsed_time:.2f}ms, which is more than 100ms")
    
    def test_dynamic_func_coef(self):
        """Test that dynamic_func_coef correctly modulates the diffusion image"""
        test_coef = 2.0
        img_mask = np.array([[0.6, 0.4], [0.3, 0.8]], dtype=np.float32)
        img_camera = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
        img_diffusion = np.array([[2.0, 2.0], [2.0, 2.0]], dtype=np.float32)
        expected = np.where(img_mask > 0.5, img_diffusion * test_coef, img_camera).astype(np.float32)
        result = process(img_camera, img_mask, img_diffusion, dynamic_func_coef=test_coef)
        np.testing.assert_array_almost_equal(result, expected)
    
    
if __name__ == "__main__":
    unittest.main()
