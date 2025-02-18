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
import datetime
import shutil
import os


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

        self.task_static = f"Write a class that derives from {self.fn_base_class}, from where you also gather insights about the range of the variables. You name the class you create DynamicClass. Critically it has to pass the existing tests in {self.fn_test}. You don't need to implement any further tests than this one."

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
            self.dynamic_processor = self.dynamic_module.DynamicClass()
            self.module_hash = current_hash
            self._backup_dynamic_class()  # Create backup when module changes
        if self.dynamic_module and self.dynamic_processor:
            x = self.dynamic_processor.process(img_camera, img_mask_segmentation, img_diffusion, dynamic_func_coef=dynamic_func_coef)
            return np.flip(x, axis=1)
        else:
            return img_camera

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
        config_override["general"] = {"plausibility_test": False, "test_path": self.fp_test}

        # Remove the function file if it exists
        if self.remove_existing_file:
            if os.path.exists(self.fp_func):
                os.remove(self.fp_func)

        # Create executor with the protoblock and config override
        executor = ProtoBlockExecutor(protoblock=self.protoblock, config_override=config_override, codebase="")

        # Execute the block (this will run tests, make changes, etc.)
        executor.execute_block()

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
            backup_files = [
                f for f in os.listdir(backup_dir)
                if f.startswith("backup_dynamic_class_") and f.endswith(".py")
            ]
            if not backup_files:
                print("No backup files found.")
                return False
            latest_backup = max(
                backup_files,
                key=lambda f: os.path.getmtime(os.path.join(backup_dir, f))
            )
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
    def delete_fn_func(self):
        """Delete or reset the fn_func attribute."""
        if hasattr(self, 'fn_func'):
            del self.fn_func
    
if __name__ == "__main__":
    import numpy as np
    import time

    processor = DynamicProcessor()
    processor.update_protoblock()
