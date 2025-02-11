import importlib.util
import hashlib
import os
from tac.protoblock.protoblock import ProtoBlock
from tac.protoblock.factory import ProtoBlockFactory

class DynamicProcessor:
    def __init__(self):
        self.module_hash = None
        self.fp_func = "/home/lugo/git/tmp/dynamic_module.py"
        self.fp_test = "/home/lugo/git/tmp/test_dynamic_module.py"
        self.fp_proto = "/home/lugo/git/tmp/bubu.json"
        self.factory = ProtoBlockFactory()


    def compute_effect(self, img_camera, img_mask_segmentation, img_diffusion):
        module_path = os.path.expanduser(self.fp_func)
        current_hash = self._compute_file_hash(module_path)

        if self.module_hash is None or self.module_hash != current_hash:
            spec = importlib.util.spec_from_file_location("dynamic_module", module_path)
            dynamic_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(dynamic_module)
            self.module_hash = current_hash

            img_camera = dynamic_module.compute_effect(img_camera, img_mask_segmentation, img_diffusion)

        return img_camera

    def _compute_file_hash(self, file_path: str) -> str:
        """Compute SHA-256 hash of file contents"""
        with open(file_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
        
    def generate_protoblock(self):
        task_description = "implement a function that computes the effect of a camera on a diffusion model"
        test_specification = "no further tests need to be written" 
        test_data_generation =  "no test data is required"
        write_files = [self.fp_func]
        context_files = [self.fp_test]
        commit_message = "None"
        test_results = None
        
        pb = ProtoBlock(task_description, test_specification, test_data_generation, write_files, context_files, commit_message, test_results)
        self.factory.save_protoblock(pb, self.fp_proto)

        
        
if __name__ == "__main__":
    import numpy as np
    import time
    


    processor = DynamicProcessor()
    processor.generate_protoblock()


    img_camera = np.random.rand(64,64,3).astype(np.float32)
    img_mask_segmentation = np.random.rand(64,64,3).astype(np.float32)
    img_diffusion = np.random.rand(64,64,3).astype(np.float32)

    with open(os.path.expanduser('~/tmp/dynamic_module.py'), 'w') as f: 
        f.write("def compute_effect(a,b,c):\n  print('Code state A')\n  return c\n")

    img_camera = processor.compute_effect(img_camera, img_mask_segmentation, img_diffusion)


    time.sleep(1)

    with open(os.path.expanduser('~/tmp/dynamic_module.py'), 'w') as f: 
        f.write("def compute_effect(a,b,c):\n  print('Code state B')\n  return c\n")

    img_camera = processor.compute_effect(img_camera, img_mask_segmentation, img_diffusion)

    