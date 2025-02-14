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
import unittest
import numpy as np
import time
from dynamic_module import DynamicClass

class DynamicClassTests(unittest.TestCase):
    def setUp(self):
        self.shape = (1024, 1024, 3)
        self.img_camera = np.ones(self.shape, dtype=np.float32)
        self.img_mask = np.zeros(self.shape, dtype=np.float32)
        self.img_diffusion = np.full(self.shape, 0.5, dtype=np.float32)
        self.dynamic_coef = 0.5
        self.processor = DynamicClass()

    def test_output_shape(self):
        result = self.processor.process(self.img_camera, self.img_mask, self.img_diffusion, self.dynamic_coef)
        self.assertEqual(result.shape, self.img_camera.shape)

    def test_data_type(self):
        result = self.processor.process(self.img_camera, self.img_mask, self.img_diffusion, self.dynamic_coef)
        self.assertEqual(result.dtype, np.float32)

    def test_performance(self):
        start_time = time.time()
        self.processor.process(self.img_camera, self.img_mask, self.img_diffusion, self.dynamic_coef)
        elapsed_time = (time.time() - start_time) * 1000
        self.assertLess(elapsed_time, 100, f"Processing took {elapsed_time:.2f}ms (>100ms)")

if __name__ == "__main__":
    unittest.main()
import unittest
import numpy as np
import time
from dynamic_module import DynamicClass

class TestDynamicClass(unittest.TestCase):
    def setUp(self):
        # Create sample test data with known shapes and float32 data type
        self.shape = (1024, 1024, 3)  # Example shape for RGB image
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
        elapsed_time = (time.time() - start_time) * 1000  # Convert to ms
        self.assertLess(elapsed_time, 100, f"Processing took {elapsed_time:.2f}ms (>100ms)")

    def test_smoothing(self):
        """Test that the smoothing buffer actually computes an exponential moving average"""
        # First call initializes the buffer with ones
        self.processor.process(np.ones(self.shape, dtype=np.float32), self.img_mask, self.img_diffusion, self.dynamic_coef)
        # Second call with zeros should average 1 and 0 based on the coef
        result = self.processor.process(np.zeros(self.shape, dtype=np.float32), self.img_mask, self.img_diffusion, self.dynamic_coef)
        # Expected calculation: new_buffer = 0.5 * ones + 0.5 * zeros = 0.5 everywhere.
        expected = np.full(self.shape, 0.5, dtype=np.float32)
        np.testing.assert_allclose(result, expected, atol=1e-6)

if __name__ == "__main__":
    unittest.main()
import unittest
import numpy as np
import time
from dynamic_module import DynamicClass

class TestDynamicClassTests(unittest.TestCase):
    def setUp(self):
        # Create sample test data: a 10x10 RGB image for simpler testing
        self.shape = (10, 10, 3)
        # Camera image: use a constant nonzero value for visible regions
        self.img_camera = np.full(self.shape, 100, dtype=np.float32)
        # Mask: set half of the image as front (1) and half as background (0)
        self.img_mask_segmentation = np.zeros(self.shape, dtype=np.uint8)
        self.img_mask_segmentation[:, :5, :] = 1
        # Diffusion image is not used in computation; use dummy data
        self.img_diffusion = np.full(self.shape, 50, dtype=np.float32)
        self.dynamic_coef = 0.5
        self.processor = DynamicClass()

    def test_output_shape(self):
        """Test that output shape matches input shape"""
        result = self.processor.process(self.img_camera, self.img_mask_segmentation, self.img_diffusion, self.dynamic_coef)
        self.assertEqual(result.shape, self.img_camera.shape)

    def test_data_type(self):
        """Test that output data type is float32"""
        result = self.processor.process(self.img_camera, self.img_mask_segmentation, self.img_diffusion, self.dynamic_coef)
        self.assertEqual(result.dtype, np.float32)

    def test_background_removal(self):
        """Test that background pixels (mask==0) are removed (set to 0)"""
        result = self.processor.process(self.img_camera, self.img_mask_segmentation, self.img_diffusion, self.dynamic_coef)
        # Regions where mask is 0 should result in 0
        background = result[:, 5:, :]
        self.assertTrue(np.all(background == 0))
        # Regions where mask is 1 should be scaled properly
        front = result[:, :5, :]
        self.assertTrue(np.all(front == 100 * self.dynamic_coef))

    def test_performance(self):
        """Test that processing time is under 100ms"""
        start_time = time.time()
        self.processor.process(self.img_camera, self.img_mask_segmentation, self.img_diffusion, self.dynamic_coef)
        elapsed_time = (time.time() - start_time) * 1000  # in ms
        self.assertLess(elapsed_time, 100, f"Processing took {elapsed_time:.2f}ms (>100ms)")

if __name__ == '__main__':
    unittest.main()
