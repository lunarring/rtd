import unittest
import numpy as np
import time
import os
import sys

# Add parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dynamic_module import process, DynamicProcessorClass


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

    def test_dynamic_class_stateful(self):
        """Test the stateful behavior of DynamicProcessorClass"""
        processor_instance = DynamicProcessorClass()
        initial_counter = processor_instance.counter
        result1 = processor_instance.process(self.img_camera, self.img_mask, self.img_diffusion, 0.5)
        self.assertEqual(result1.shape, self.img_camera.shape)
        self.assertEqual(result1.dtype, self.img_camera.dtype)
        self.assertEqual(processor_instance.counter, initial_counter + 1)
        result2 = processor_instance.process(self.img_camera, self.img_mask, self.img_diffusion, 0.5)
        self.assertEqual(processor_instance.counter, initial_counter + 2)


if __name__ == "__main__":
    unittest.main()
