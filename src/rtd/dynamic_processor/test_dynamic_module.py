import unittest
import numpy as np
import time
from dynamic_module import process

class TestDynamicModule(unittest.TestCase):
    def setUp(self):
        # Create sample test data with known shapes
        self.shape = (256, 256, 3)  # Example shape for RGB image
        self.img_camera = np.ones(self.shape)
        self.img_mask = np.zeros(self.shape)
        self.img_diffusion = np.full(self.shape, 0.5)

    def test_output_shape(self):
        """Test that output shape matches input shape"""
        result = process(self.img_camera, self.img_mask, self.img_diffusion)
        self.assertEqual(result.shape, self.img_camera.shape)

    def test_data_type(self):
        """Test that output data type matches input data type"""
        result = process(self.img_camera, self.img_mask, self.img_diffusion)
        self.assertEqual(result.dtype, self.img_camera.dtype)

    def test_first_input_recovery(self):
        """Test that we can recover first input by subtracting others"""
        result = process(self.img_camera, self.img_mask, self.img_diffusion)
        recovered = result - self.img_mask - self.img_diffusion
        np.testing.assert_array_equal(recovered, self.img_camera)

    def test_performance(self):
        """Test that processing time is under 100ms"""
        start_time = time.time()
        process(self.img_camera, self.img_mask, self.img_diffusion)
        elapsed_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        self.assertLess(elapsed_time, 100, f"Processing took {elapsed_time:.2f}ms, which is more than 100ms")

if __name__ == '__main__':
    unittest.main()
