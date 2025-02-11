import numpy as np
from src.rtd.utils.input_image import InputImageProcessor

def test_input_image_processor_with_segmentation():
    # Generate synthetic image data: 256x256x3 with uint8 values.
    dummy_img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    processor = InputImageProcessor()  # segmentation enabled by default
    processor.set_blur_size(3)
    processor.set_brightness(1.2)
    processor.set_flip(True, flip_axis=0)
    
    processed_img, seg_mask = processor.process(dummy_img)
    
    # Verify processed image is a numpy array with same shape and dtype as input.
    assert isinstance(processed_img, np.ndarray), "Processed image is not a numpy array."
    assert processed_img.shape == dummy_img.shape, f"Expected shape {dummy_img.shape}, got {processed_img.shape}"
    assert processed_img.dtype == np.uint8, f"Expected dtype uint8, got {processed_img.dtype}"
    
    # If segmentation mask is provided, ensure its spatial dimensions match the input.
    if seg_mask is not None:
        assert seg_mask.shape[0] == dummy_img.shape[0] and seg_mask.shape[1] == dummy_img.shape[1], (
            f"Segmentation mask dimensions {seg_mask.shape[:2]} do not match input image dimensions {dummy_img.shape[:2]}"
        )

def test_input_image_processor_without_segmentation():
    # Generate synthetic image data.
    dummy_img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    processor = InputImageProcessor()
    processor.set_human_seg(False)  # Disable segmentation.
    processor.set_blur_size(3)
    
    processed_img, seg_mask = processor.process(dummy_img)
    
    # Verify processed image is a numpy array with same shape and dtype as input.
    assert isinstance(processed_img, np.ndarray), "Processed image is not a numpy array."
    assert processed_img.shape == dummy_img.shape, f"Expected shape {dummy_img.shape}, got {processed_img.shape}"
    assert processed_img.dtype == np.uint8, f"Expected dtype uint8, got {processed_img.dtype}"
    
    # With segmentation disabled, seg_mask should be None.
    assert seg_mask is None, "Segmentation mask should be None when segmentation is disabled."

if __name__ == '__main__':
    test_input_image_processor_with_segmentation()
    test_input_image_processor_without_segmentation()
    print("All tests passed.")
    
