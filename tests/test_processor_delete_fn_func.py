import os
import unittest
from src/rtd/dynamic_processor.processor_dynamic_module import DynamicProcessor

class TestDeleteCurrentFnFunc(unittest.TestCase):
    def test_delete_current_fn_func(self):
        dp = DynamicProcessor()
        # Write temporary marker content to dynamic module file
        with open(dp.fp_func, "w") as f:
            f.write("dummy content")
        # Verify file is not empty
        self.assertGreater(os.path.getsize(dp.fp_func), 0)
        # Invoke the delete method
        dp.delete_current_fn_func()
        # Verify file has been emptied
        self.assertEqual(os.path.getsize(dp.fp_func), 0)

if __name__ == "__main__":
    unittest.main()
