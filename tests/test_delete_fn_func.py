import unittest
from src.rtd.dynamic_processor.processor_dynamic_module import DynamicProcessor

class TestDeleteFnFunc(unittest.TestCase):
    def test_delete_fn_func(self):
        dp = DynamicProcessor()
        # Assign a dummy function to fn_func
        dp.fn_func = lambda x: x
        self.assertTrue(hasattr(dp, 'fn_func'))
        dp.delete_fn_func()
        self.assertFalse(hasattr(dp, 'fn_func'))

if __name__ == '__main__':
    unittest.main()
