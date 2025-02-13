import importlib.util
import hashlib
import os
from tac.protoblock.protoblock import ProtoBlock
from tac.protoblock.factory import ProtoBlockFactory
from tac.core.executor import ProtoBlockExecutor


class DynamicProcessor:
    def __init__(self, base_dir=None):
        self.module_hash = None
        # Use default base_dir if none provided
        self.base_dir = "."

        # Assemble full paths
        self.fp_func = os.path.join(self.base_dir, "dynamic_module.py")
        self.fp_test = os.path.join(self.base_dir, "tests/test_dynamic_module.py")
        self.fp_proto = os.path.join(self.base_dir, "module.json")

        self.factory = ProtoBlockFactory()
        self.protoblock = None

    def compute_effect(self, img_camera, img_mask_segmentation, img_diffusion):
        module_path = os.path.expanduser(self.fp_func)
        current_hash = self._compute_file_hash(module_path)

        if self.module_hash is None or self.module_hash != current_hash:
            spec = importlib.util.spec_from_file_location("dynamic_module", module_path)
            dynamic_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(dynamic_module)
            self.module_hash = current_hash

            img_camera = dynamic_module.compute_effect(
                img_camera, img_mask_segmentation, img_diffusion
            )

        return img_camera

    def _compute_file_hash(self, file_path: str) -> str:
        """Compute SHA-256 hash of file contents"""
        with open(file_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    def generate_protoblock(self, task_description):
        # Delete existing protoblock file if it exists
        if os.path.exists(self.fp_proto):
            os.remove(self.fp_proto)

        task_description = task_description
        test_specification = ""
        test_data_generation = ""
        write_files = [self.fp_func]
        context_files = [self.fp_test]
        commit_message = "None"
        test_results = None

        protoblock = ProtoBlock(
            task_description,
            test_specification,
            test_data_generation,
            write_files,
            context_files,
            commit_message,
            test_results,
        )
        self.factory.save_protoblock(protoblock, self.fp_proto)
        self.protoblock = protoblock
        return protoblock

    def execute_protoblock(self):
        # Create config override dictionary to disable git and plausibility check
        config_override = {}
        config_override["git"] = {"enabled": False}
        config_override["general"] = {"plausibility_test": False}

        # Create executor with the protoblock and config override
        executor = ProtoBlockExecutor(
            protoblock=self.protoblock, config_override=config_override, codebase=""
        )

        # Execute the block (this will run tests, make changes, etc.)
        executor.execute_block()


if __name__ == "__main__":
    import numpy as np
    import time

    processor = DynamicProcessor()

    task_user = "let us make an interesting function that takes the camera images and adds it to itself flipped left to right, and ensure that the range of values is the same as in the input images."
    task_static = "the input of the function are three numpy arrays that are images: img_camera, img_mask_segmentation, img_diffusion, which are all float32. we need one numpy array as output. it has to pass the existing test. make sure the function name is called 'process'."
    task_description = task_user + "\n" + task_static

    protoblock = processor.generate_protoblock(task_description)

    processor.execute_protoblock()

    # # Execute tac make command
    # import subprocess
    # import sys

    # # Construct the conda run command to ensure we're in rtd environment
    # cmd = "source ~/.bashrc; conda init; conda activate rtd; tac make --json module.json --no-git --plausibility-check false"

    # try:
    #     result = subprocess.run(cmd, shell=True, check=True, executable="/bin/bash", capture_output=True, text=True)
    #     print("Command output:")
    #     print(result.stdout)
    #     if result.stderr:
    #         print("Errors/Warnings:")
    #         print(result.stderr)
    # except subprocess.CalledProcessError as e:
    #     print(f"Error executing command: {e}")
    #     print("Error output:")
    #     print(e.stderr)
    #     sys.exit(1)

    # xxx

    # img_camera = np.random.rand(64,64,3).astype(np.float32)
    # img_mask_segmentation = np.random.rand(64,64,3).astype(np.float32)
    # img_diffusion = np.random.rand(64,64,3).astype(np.float32)

    # with open(os.path.expanduser('~/tmp/dynamic_module.py'), 'w') as f:
    #     f.write("def compute_effect(a,b,c):\n  print('Code state A')\n  return c\n")

    # img_camera = processor.compute_effect(img_camera, img_mask_segmentation, img_diffusion)

    # time.sleep(1)

    # with open(os.path.expanduser('~/tmp/dynamic_module.py'), 'w') as f:
    #     f.write("def compute_effect(a,b,c):\n  print('Code state B')\n  return c\n")

    # img_camera = processor.compute_effect(img_camera, img_mask_segmentation, img_diffusion)
