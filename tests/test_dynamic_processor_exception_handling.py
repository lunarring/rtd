import numpy as np
import importlib.util
import os
import pytest
from src.rtd.dynamic_processor.processor_dynamic_module import DynamicProcessor

def fake_exec_module(module):
    raise Exception("Simulated dynamic module failure")

def test_exception_handling(monkeypatch):
    processor = DynamicProcessor()

    # Patch importlib.util.spec_from_file_location to simulate a failure during exec_module
    original_spec_from_file_location = importlib.util.spec_from_file_location

    def fake_spec_from_file_location(name, location):
        spec = original_spec_from_file_location(name, location)
        spec.loader.exec_module = lambda module: fake_exec_module(module)
        return spec

    monkeypatch.setattr(importlib.util, "spec_from_file_location", fake_spec_from_file_location)

    # Create dummy images (all-black)
    dummy_cam = np.zeros((100, 100, 3), dtype=np.uint8)
    dummy_mask = np.zeros((100, 100, 3), dtype=np.uint8)
    dummy_diff = np.zeros((100, 100, 3), dtype=np.uint8)

    result = processor.process(dummy_cam, dummy_mask, dummy_diff)

    # Verify that the fallback image has a prominent green stripe in the center.
    h, w, c = result.shape
    stripe_width = max(1, w // 5)
    center = w // 2
    start = max(0, center - stripe_width // 2)
    end = min(w, start + stripe_width)
    stripe = result[:, start:end]

    # Check that each pixel in the stripe is green (RGB=[0,255,0])
    assert np.all(stripe[:, :, 0] == 0)
    assert np.all(stripe[:, :, 1] == 255)
    assert np.all(stripe[:, :, 2] == 0)
