import numpy as np
import torch
import pytest
from src.rtd.dynamic_processor.dynamic_module import DynamicClass

def test_cuda_processing():
    # Create dummy images of size 256x256 with 3 channels.
    dummy_camera = np.random.randint(0, 256, size=(256, 256, 3), dtype=np.uint8)
    dummy_mask = np.ones((256, 256, 3), dtype=np.uint8)  # segmentation mask as ones
    dummy_diffusion = np.random.randint(0, 256, size=(256, 256, 3), dtype=np.uint8)
    
    dynamic_instance = DynamicClass()
    # Verify that the device is set to cuda
    assert dynamic_instance.device.type == "cuda"
    
    output = dynamic_instance.process(dummy_camera, dummy_mask, dummy_diffusion, 1.0)
    
    # Check that the output is a numpy array with the same spatial dimensions as the input
    assert isinstance(output, np.ndarray)
    assert output.shape == dummy_camera.shape
def test_cuda_processing_torch_inputs():
    import torch
    from src.rtd.dynamic_processor.dynamic_module import DynamicClass
    device = torch.device("cuda")
    dummy_camera = (torch.rand(256, 256, 3, device=device) * 255).float()
    dummy_mask = torch.ones(256, 256, 3, device=device).float()
    dummy_diffusion = (torch.rand(256, 256, 3, device=device) * 255).float()
    
    dynamic_instance = DynamicClass()
    # Verify that the device is set to cuda
    assert dynamic_instance.device.type == "cuda"

    output = dynamic_instance.process(dummy_camera, dummy_mask, dummy_diffusion, 1.0)

    # Check that the output is a torch tensor with the same spatial dimensions as the input and on CUDA.
    assert isinstance(output, torch.Tensor)
    assert output.shape == dummy_camera.shape
    assert output.device.type == "cuda"
