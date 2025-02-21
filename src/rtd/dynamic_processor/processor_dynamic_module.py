import numpy as np
import lunar_tools as lt
import importlib.util
import importlib.resources
import hashlib
import os
from tac.protoblock.protoblock import ProtoBlock
from tac.protoblock.factory import ProtoBlockFactory
from tac.core.block_runner import BlockRunner
import rtd.dynamic_processor
from tac.cli.voice import VoiceUI
import datetime
import shutil
import os
import torch


class DynamicProcessor:
    def __init__(self):
        self.module_hash = None
        # Use default base_dir if none provided
        # self.base_dir = "."

        self.fn_func = "dynamic_module.py"
        self.fn_test = "test_dynamic_class.py"
        self.fn_base_class = "base_dynamic_module.py"

        # Use importlib.resources to get package paths
        with importlib.resources.path("rtd.dynamic_processor", self.fn_func) as p:
            self.fp_func = str(p)
        with importlib.resources.path("rtd.dynamic_processor.tests", self.fn_test) as p:
            self.fp_test = str(p)
        with importlib.resources.path("rtd.dynamic_processor", "dynamic_module.json") as p:
            self.fp_proto = str(p)
        with importlib.resources.path("rtd.dynamic_processor", self.fn_base_class) as p:
            self.fp_base_class = str(p)

        self.factory = ProtoBlockFactory()
        self.protoblock = None
        self.dynamic_module = None
        self.dynamic_processor = None
        self.remove_existing_file = False

        self.task_static = f"Write a class that derives from a base class {self.fn_base_class}, from where you also gather insights about the range of the variables. You name the class you create DynamicClass. Critically it has to pass the existing tests in {self.fn_test}. Always make sure to at least pass all the variables that are listed in the base class, particularly the list of dynamic_func_coef. You don't need to implement any further tests than this one."

    def process(self, img_camera, img_mask_segmentation, img_diffusion, img_optical_flow, dynamic_coef):
        if not os.path.exists(self.fp_func):
            return img_camera
        # if list_dynamic_coef is None:
        #     list_dynamic_coef = [0.5]

        img_camera = torch.tensor(np.asarray(img_camera), device="cuda")
        img_mask_segmentation = torch.tensor(np.asarray(img_mask_segmentation), device="cuda")
        img_mask_segmentation = torch.flip(img_mask_segmentation, dims=[1])
        img_diffusion = torch.tensor(np.asarray(img_diffusion), device="cuda")
        img_optical_flow = torch.tensor(np.asarray(img_optical_flow), device="cuda")

        img_camera = lt.resize(img_camera, size=(img_diffusion.shape[0], img_diffusion.shape[1]))
        img_optical_flow = lt.resize(img_optical_flow, size=(img_diffusion.shape[0], img_diffusion.shape[1]))
        img_mask_segmentation = lt.resize(img_mask_segmentation, size=(img_diffusion.shape[0], img_diffusion.shape[1]))
        # print(f"img_camera.shape: {img_camera.shape}")
        # print(f"img_mask_segmentation.shape: {img_mask_segmentation.shape}")
        # print(f"img_diffusion.shape: {img_diffusion.shape}")
        # print(f"img_optical_flow.shape: {img_optical_flow.shape}")

        try:
            # Use the already resolved path from importlib.resources
            current_hash = self._compute_file_hash(self.fp_func)
            if self.module_hash is None or self.module_hash != current_hash:
                spec = importlib.util.spec_from_file_location("dynamic_module", self.fp_func)
                self.dynamic_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(self.dynamic_module)
                self.dynamic_processor = self.dynamic_module.DynamicClass()
                self.module_hash = current_hash
                self._backup_dynamic_class()  # Create backup when module changes
                print("Dynamic module changed, reloading")
            if self.dynamic_module and self.dynamic_processor:
                x = self.dynamic_processor.process(img_camera, img_mask_segmentation, img_diffusion, img_optical_flow, dynamic_coef)
                return torch.flip(x, dims=[1]).cpu().numpy()
            else:
                raise Exception("Dynamic Processor not available")
        except Exception as e:
            print(f"dynamic module reloading failed: {e}")
            fallback = img_camera.cpu().numpy()
            if fallback.ndim == 3 and fallback.shape[2] >= 3:
                h, w, c = fallback.shape
                stripe_width = max(1, w // 5)
                center = w // 2
                start = max(0, center - stripe_width // 2)
                end = min(w, start + stripe_width)
                fallback[:, start:end] = [0, 255, 0]
            return fallback

    def _compute_file_hash(self, file_path: str) -> str:
        """Compute SHA-256 hash of file contents"""
        with open(file_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    def _backup_dynamic_class(self):
        """Create a backup of the current dynamic class file with timestamp."""
        # Create backup directory if it doesn't exist
        with importlib.resources.path("rtd.dynamic_processor", "") as p:
            backup_dir = os.path.join(str(p), "backup")
        os.makedirs(backup_dir, exist_ok=True)

        # Generate timestamp
        timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M")
        backup_filename = f"backup_dynamic_class_{timestamp}.py"
        backup_path = os.path.join(backup_dir, backup_filename)

        # Copy the file
        shutil.copy2(self.fp_func, backup_path)

    def generate_protoblock(self, task_description):
        # Delete existing protoblock file if it exists
        if os.path.exists(self.fp_proto):
            os.remove(self.fp_proto)

        task_description = task_description
        test_specification = ""
        test_data_generation = ""
        write_files = [self.fp_func]
        context_files = [self.fp_test, self.fp_base_class]
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
        config_override["general"] = {"plausibility_test": False, "test_path": self.fp_test, "type": "native", "reasoning_effort": "medium"}

        # Remove the function file if it exists
        if self.remove_existing_file:
            if os.path.exists(self.fp_func):
                os.remove(self.fp_func)

        block_runner = BlockRunner(json_file=self.fp_proto, config_override=config_override)
        return block_runner.run_loop()

    def update_protoblock(self, task_user=None):
        if task_user is None:
            task_user = "let us just use the human segmentation mask and fill it with noise"
        task_description = task_user + "\n" + self.task_static

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

        task_description = task_user + "\n" + self.task_static

        self.generate_protoblock(task_description)
        self.execute_protoblock()

        # Inject completion message
        voice_ui.inject_message("I have completed the programming task according to your instructions.")

    def restore_backup(self):
        """Restore the dynamic module file from the latest backup."""
        try:
            with importlib.resources.path("rtd.dynamic_processor", "") as p:
                backup_dir = os.path.join(str(p), "backup")
            backup_files = [f for f in os.listdir(backup_dir) if f.startswith("backup_dynamic_class_") and f.endswith(".py")]
            if not backup_files:
                print("No backup files found.")
                return False
            latest_backup = max(backup_files, key=lambda f: os.path.getmtime(os.path.join(backup_dir, f)))
            backup_path = os.path.join(backup_dir, latest_backup)
            with open(backup_path, "rb") as bf:
                backup_content = bf.read()
            with open(self.fp_func, "wb") as df:
                df.write(backup_content)
            print(f"Restored dynamic module from {latest_backup}")
            return True
        except Exception as e:
            print(f"Failed to restore backup: {e}")
            return False

    def delete_current_fn_func(self):
        """Empties the dynamic module file referenced by fn_func."""
        with open(self.fp_func, "w") as f:
            f.write("")


if __name__ == "__main__":
    import numpy as np
    import time

    processor = DynamicProcessor()
    processor.update_protoblock()
