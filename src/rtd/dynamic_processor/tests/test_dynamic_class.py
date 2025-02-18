import unittest
import numpy as np
import time
import torch
from rtd.dynamic_processor.base_dynamic_module import BaseDynamicClass
from dynamic_module import DynamicClass

class TestDynamicModule(DynamicClass):
    """Concrete implementation of BaseDynamicClass for testing using DynamicClass implementation."""
    pass

class TestDynamicModuleTests(unittest.TestCase):
    def setUp(self):
        # Create sample test data with known shapes and float32 data type
        self.shape = (1024, 1024, 3)  # Example shape for RGB image
        self.img_camera = torch.ones(self.shape, dtype=torch.float32)
        self.img_mask = torch.zeros(self.shape, dtype=torch.float32)
        self.img_diffusion = torch.full(self.shape, 0.5, dtype=torch.float32)
        self.dynamic_coef = 0.5
        self.processor = TestDynamicModule()

    def test_output_shape(self):
        """Test that output shape matches input shape"""
        result = self.processor.process(self.img_camera, self.img_mask, self.img_diffusion, self.dynamic_coef)
        self.assertEqual(result.shape, self.img_camera.shape)

    def test_data_type(self):
        """Test that output data type is float32"""
        result = self.processor.process(self.img_camera, self.img_mask, self.img_diffusion, self.dynamic_coef)
        self.assertEqual(result.dtype, torch.float32)

    def test_performance(self):
        """Test that processing time is under 100ms"""
        start_time = time.time()
        self.processor.process(self.img_camera, self.img_mask, self.img_diffusion, self.dynamic_coef)
        elapsed_time = (time.time() - start_time) * 1000  # ms
        self.assertLess(elapsed_time, 100, f"Processing took {elapsed_time:.2f}ms (>100ms)")

    def test_segmentation_mask(self):
        """Test that the segmentation mask correctly zeroes out the output where mask is 0."""
        # Create a camera image with all values equal to 2
        img_camera = torch.full(self.shape, 2, dtype=torch.float32)
        # Create a segmentation mask that is 1 in the top half and 0 in the bottom half
        img_mask = torch.zeros(self.shape, dtype=torch.float32)
        img_mask[:512, :, :] = 1.0
        result = self.processor.process(img_camera, img_mask, self.img_diffusion, 1.0)
        # Check that the bottom half is all zeros
        self.assertTrue(torch.all(result[512:, :, :] == 0).item(), "Bottom half of result should be zero due to mask.")
        # Check that the top half is not all zero
        self.assertFalse(torch.all(result[:512, :, :] == 0).item(), "Top half of result should not be zero.")

    def test_output_range(self):
        """Test that the output values are within the valid image range [0, 255]."""
        # Test with extreme input values using torch tensors
        img_camera = torch.rand(self.shape, dtype=torch.float32) * 300
        result = self.processor.process(img_camera, self.img_mask, self.img_diffusion, self.dynamic_coef)
        
        # Check that all values are within [0, 255]
        self.assertTrue(torch.all(result >= 0).item(), "Output contains values less than 0")
        self.assertTrue(torch.all(result <= 255).item(), "Output contains values greater than 255")

if __name__ == "__main__":
    unittest.main()
