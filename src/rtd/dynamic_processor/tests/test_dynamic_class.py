import unittest
import time
import torch
from rtd.dynamic_processor.base_dynamic_module import BaseDynamicClass
from dynamic_module import DynamicClass


class TestDynamicModuleTests(unittest.TestCase):
    def setUp(self):
        # Create sample test data with known shapes and float32 data type
        self.shape_3 = (768, 1024, 3)  # Example shape for RGB image
        self.shape_2 = (768, 1024, 2)  # Example shape for RGB image
        self.img_mask = torch.zeros(self.shape_3, dtype=torch.float32, device="cuda")
        self.img_diffusion = torch.full(self.shape_3, 0.5, dtype=torch.float32, device="cuda")
        self.img_optical_flow = torch.full(self.shape_2, 0.5, dtype=torch.float32, device="cuda")
        self.dynamic_coef = 0.5  # Single coefficient
        self.processor = DynamicClass()

    def test_output_shape(self):
        """Test that output shape matches input shape"""
        result = self.processor.process(self.img_diffusion, self.img_mask, self.img_optical_flow, self.dynamic_coef)
        self.assertEqual(result.shape, self.img_diffusion.shape)

    def test_data_type(self):
        """Test that output data type is float32"""
        result = self.processor.process(self.img_diffusion, self.img_mask, self.img_optical_flow, self.dynamic_coef)
        self.assertEqual(result.dtype, torch.float32)

    def test_performance(self):
        """Test that processing time is under 100ms"""
        start_time = time.time()
        result = self.processor.process(self.img_diffusion, self.img_mask, self.img_optical_flow, self.dynamic_coef)
        elapsed_time = (time.time() - start_time) * 1000  # ms
        msg = f"Processing took {elapsed_time:.2f}ms (>100ms)"
        self.assertLess(elapsed_time, 100, msg)

    def test_output_range(self):
        """Test output values are within valid image range [0, 255]."""
        # Test with extreme input values using torch tensors
        result = self.processor.process(self.img_diffusion, self.img_mask, self.img_optical_flow, self.dynamic_coef)

        # Check that all values are within [0, 255]
        msg_low = "Output contains values less than 0"
        msg_high = "Output contains values greater than 255"
        self.assertTrue(torch.all(result >= 0).item(), msg_low)
        self.assertTrue(torch.all(result <= 255).item(), msg_high)


if __name__ == "__main__":
    unittest.main()
