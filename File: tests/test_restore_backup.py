import os
import shutil
import hashlib
import tempfile
import unittest

from src.rtd.dynamic_processor.processor_dynamic_module import DynamicProcessor

class TestRestoreBackup(unittest.TestCase):
    def setUp(self):
        self.processor = DynamicProcessor()
        self.dynamic_module_path = self.processor.fp_func

        # Backup the original dynamic module file temporarily
        self.temp_original = tempfile.NamedTemporaryFile(delete=False)
        self.temp_original.close()
        shutil.copy2(self.dynamic_module_path, self.temp_original.name)

        # Simulate a modification by appending a temporary string
        with open(self.dynamic_module_path, "a") as f:
            f.write("\n# Temporary modification for testing restore_backup\n")

    def tearDown(self):
        # Restore the original dynamic module file from our temporary copy
        shutil.copy2(self.temp_original.name, self.dynamic_module_path)
        os.unlink(self.temp_original.name)

    def _compute_file_hash(self, file_path):
        with open(file_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    def test_restore_backup(self):
        # Call restore_backup to restore the backup file to the dynamic module file
        status = self.processor.restore_backup()
        self.assertTrue(status, "restore_backup did not return success")

        # Compute the hash of the restored dynamic module file
        hash_restored = self._compute_file_hash(self.dynamic_module_path)

        # Determine the backup file path:
        backup_dir = os.path.join(os.path.dirname(self.dynamic_module_path), "backup")
        backup_file = os.path.join(backup_dir, "backup_dynamic_class_250214_0926.py")
        self.assertTrue(os.path.exists(backup_file), "Expected backup file does not exist")

        hash_backup = self._compute_file_hash(backup_file)
        self.assertEqual(hash_restored, hash_backup, "Restored file hash does not match backup file hash")

if __name__ == '__main__':
    unittest.main()
