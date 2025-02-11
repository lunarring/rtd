import importlib.util
import hashlib
import os

class DynamicProcessor:
    def __init__(self):
        self.module_hash = None

    def compute_effect(self, img_camera, img_mask_segmentation, img_diffusion):
        module_path = os.path.expanduser('~/tmp/dynamic_module.py')
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
        
if __name__ == "__main__":
    import numpy as np
    import time

    processor = DynamicProcessor()
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

    