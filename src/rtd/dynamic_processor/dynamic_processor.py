import numpy as np
import lunar_tools as lt
import importlib.util
import importlib.resources
import hashlib
import os
from tac.protoblock.protoblock import ProtoBlock
from tac.protoblock.factory import ProtoBlockFactory
from tac.core.executor import ProtoBlockExecutor
import rtd.dynamic_processor
from tac.cli.voice import VoiceUI


class DynamicProcessor:
    def __init__(self, base_dir=None):
        self.module_hash = None
        # Use default base_dir if none provided
        self.base_dir = "."

        # Use importlib.resources to get package paths
        with importlib.resources.path("rtd.dynamic_processor", "dynamic_module.py") as p:
            self.fp_func = str(p)
        with importlib.resources.path("rtd.dynamic_processor.tests", "test_dynamic_module.py") as p:
            self.fp_test = str(p)
        with importlib.resources.path("rtd.dynamic_processor", "dynamic_module.json") as p:
            self.fp_proto = str(p)

        self.factory = ProtoBlockFactory()
        self.protoblock = None
        self.dynamic_module = None

    def process(self, img_camera, img_mask_segmentation, img_diffusion, dynamic_func_coef=0.5):
        if not os.path.exists(self.fp_func):
            return img_camera
        img_camera = np.asarray(img_camera)
        img_mask_segmentation = np.asarray(img_mask_segmentation)
        img_mask_segmentation = np.flip(img_mask_segmentation, axis=1)
        img_diffusion = np.asarray(img_diffusion)

        img_camera = lt.resize(img_camera, size=(img_diffusion.shape[0], img_diffusion.shape[1]))
        img_mask_segmentation = lt.resize(img_mask_segmentation, size=(img_diffusion.shape[0], img_diffusion.shape[1]))

        # Use the already resolved path from importlib.resources
        current_hash = self._compute_file_hash(self.fp_func)

        if self.module_hash is None or self.module_hash != current_hash:
            print("Dynamic module changed, reloading")
            spec = importlib.util.spec_from_file_location("dynamic_module", self.fp_func)
            self.dynamic_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.dynamic_module)
            self.module_hash = current_hash
        if self.dynamic_module:
            x = self.dynamic_module.process(img_camera, img_mask_segmentation, img_diffusion, dynamic_func_coef=dynamic_func_coef)
            return np.flip(x, axis=1)
        else:
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

    def execute_protoblock(self):
        # Create config override dictionary to disable git and plausibility check
        config_override = {}
        config_override["git"] = {"enabled": False}
        config_override["general"] = {"plausibility_test": False, "test_path": "src/rtd/dynamic_processor/tests/test_dynamic_module.py"}

        # Remove the function file if it exists
        if os.path.exists(self.fp_func):
            os.remove(self.fp_func)

        # Create executor with the protoblock and config override
        executor = ProtoBlockExecutor(protoblock=self.protoblock, config_override=config_override, codebase="")

        # Execute the block (this will run tests, make changes, etc.)
        executor.execute_block()

    def update_protoblock(self):
        task_user = "let us just use the human segmentation mask and fill it with noise"
        task_static = "the input of the function are three numpy arrays that are images: img_camera, img_mask_segmentation, img_diffusion, which are all float32. also we get a float parameter [0,1] called dynamic_func_coef.we need one numpy array as output. it has to pass the existing test. make sure the function name is called 'process'."
        task_description = task_user + "\n" + task_static

        self.generate_protoblock(task_description)
        self.execute_protoblock()

    def update_protoblock_voice(self):
        # Initialize voice UI
        voice_ui = VoiceUI()
        voice_ui.start()

        # Wait for voice instructions
        task_user = voice_ui.wait_until_prompt()
        print(f"got task_user: {task_user}")

        # Inject confirmation message
        voice_ui.inject_message("I understand your request. I will now start programming according to your instructions.")

        task_static = "the input of the function are three numpy arrays that are images and one float parameter: img_camera, img_mask_segmentation, img_diffusion, dynamic_func_coef, which are all float32. we need one numpy array as output. it has to pass the existing test. make sure the function name is called 'process'."
        task_description = task_user + "\n" + task_static

        self.generate_protoblock(task_description)
        self.execute_protoblock()

        # Inject completion message
        voice_ui.inject_message("I have completed the programming task according to your instructions.")


if __name__ == "__main__":
    import numpy as np
    import time

    processor = DynamicProcessor()
    processor.update_protoblock()
