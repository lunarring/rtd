import unittest
import numpy as np
import time
from rtd.dynamic_processor.base_dynamic_module import BaseDynamicClass


class TestDynamicModule(BaseDynamicClass):
    """Concrete implementation of BaseDynamicModule for testing."""

    def process(
        self,
        img_camera: np.ndarray,
        img_mask_segmentation: np.ndarray,
        img_diffusion: np.ndarray,
        dynamic_func_coef: float,
    ) -> np.ndarray:
        # Simple implementation for testing
        return img_camera * dynamic_func_coef


class TestDynamicModuleTests(unittest.TestCase):
    def setUp(self):
        # Create sample test data with known shapes and float32 data type
        self.shape = (1024, 1024, 3)  # Example shape for RGB image
        self.img_camera = np.ones(self.shape, dtype=np.float32)
        self.img_mask = np.zeros(self.shape, dtype=np.float32)
        self.img_diffusion = np.full(self.shape, 0.5, dtype=np.float32)
        self.dynamic_coef = 0.5
        self.processor = TestDynamicModule()

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

    def test_segmentation_mask(self):
        """Test that the segmentation mask correctly zeroes out the output where mask is 0."""
        # Create a camera image with all values equal to 2
        img_camera = np.full(self.shape, 2, dtype=np.float32)
        # Create a segmentation mask that is 1 in the top half and 0 in the bottom half
        img_mask = np.zeros(self.shape, dtype=np.float32)
        img_mask[:512, :, :] = 1.0
        result = self.processor.process(img_camera, img_mask, self.img_diffusion, 1.0)
        # Check that the bottom half is not all zeros due to circles overlay
        self.assertFalse(np.all(result[512:, :, :] == 0), "Bottom half of result should not be zero due to circles overlay.")
        # Check that the top half is not all zero
        self.assertFalse(np.all(result[:512, :, :] == 0), "Top half of result should not be zero.")

    def test_output_range(self):
        """Test that the output values are within the valid image range [0, 255]."""
        # Test with extreme input values
        img_camera = np.random.uniform(0, 300, self.shape).astype(np.float32)  # Values beyond 255
        result = self.processor.process(img_camera, self.img_mask, self.img_diffusion, self.dynamic_coef)
        
        # Check that all values are within [0, 255]
        self.assertTrue(np.all(result >= 0), "Output contains values less than 0")
        self.assertTrue(np.all(result <= 255), "Output contains values greater than 255")
    
    def test_circle_colors(self):
        """Test that each circle has a valid color attribute."""
        # Trigger initialization by calling process
        self.processor.process(self.img_camera, self.img_mask, self.img_diffusion, self.dynamic_coef)
        self.assertTrue(self.processor.initialized, "Processor should be initialized after process call.")
        for group in self.processor.circles:
            for circle in group:
                self.assertIn('color', circle, "Circle does not have a 'color' attribute.")
                color = circle['color']
                self.assertEqual(len(color), 3, "Color attribute should be a tuple of 3 elements.")
                for channel in color:
                    self.assertGreaterEqual(channel, 0, "Color channel should be >= 0")
                    self.assertLessEqual(channel, 255, "Color channel should be <= 255")


if __name__ == "__main__":
    unittest.main()
import unittest
import numpy as np
import time
from rtd.dynamic_processor.base_dynamic_module import BaseDynamicClass
from dynamic_module import DynamicClass

class TestDynamicModule(DynamicClass):
    """Concrete implementation of BaseDynamicClass for testing using DynamicClass implementation."""
    pass

class TestDynamicModuleTests(unittest.TestCase):
    def setUp(self):
        # Create sample test data with known shapes and float32 data type
        self.shape = (1024, 1024, 3)  # Example shape for RGB image
        self.img_camera = np.ones(self.shape, dtype=np.float32)
        self.img_mask = np.zeros(self.shape, dtype=np.float32)
        self.img_diffusion = np.full(self.shape, 0.5, dtype=np.float32)
        self.dynamic_coef = 0.5
        self.processor = TestDynamicModule()

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

    def test_segmentation_mask(self):
        """Test that the segmentation mask correctly zeroes out the output where mask is 0."""
        # Create a camera image with all values equal to 2
        img_camera = np.full(self.shape, 2, dtype=np.float32)
        # Create a segmentation mask that is 1 in the top half and 0 in the bottom half
        img_mask = np.zeros(self.shape, dtype=np.float32)
        img_mask[:512, :, :] = 1.0
        result = self.processor.process(img_camera, img_mask, self.img_diffusion, 1.0)
        # Check that the bottom half is all zeros
        self.assertTrue(np.all(result[512:, :, :] == 0), "Bottom half of result should be zero due to mask.")
        # Check that the top half is not all zero
        self.assertFalse(np.all(result[:512, :, :] == 0), "Top half of result should not be zero.")

    def test_output_range(self):
        """Test that the output values are within the valid image range [0, 255]."""
        # Test with extreme input values
        img_camera = np.random.uniform(0, 300, self.shape).astype(np.float32)  # Values beyond 255
        result = self.processor.process(img_camera, self.img_mask, self.img_diffusion, self.dynamic_coef)
        
        # Check that all values are within [0, 255]
        self.assertTrue(np.all(result >= 0), "Output contains values less than 0")
        self.assertTrue(np.all(result <= 255), "Output contains values greater than 255")

if __name__ == "__main__":
    unittest.main()
