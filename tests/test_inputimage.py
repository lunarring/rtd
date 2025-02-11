import numpy as np
from rtd.utils.input_image import InputImageProcessor

def test_input_image_processor():
    # Create a dummy image with dimensions 256x256x3 and dtype uint8
    dummy_img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    processor = InputImageProcessor()
    processor.set_blur_size(3)  # Ensure blur is properly set to avoid runtime errors
    processed_img, seg_mask = processor.process(dummy_img)
    # Assert that the processed image is a numpy array with the same shape as the input image
    assert isinstance(processed_img, np.ndarray), "Processed image is not a numpy array."
    assert processed_img.shape == dummy_img.shape, f"Expected shape {dummy_img.shape}, got {processed_img.shape}"
    # If seg_mask is provided, then its spatial dimensions should match the input image's height and width
    if seg_mask is not None:
        assert seg_mask.shape[:2] == dummy_img.shape[:2], "Segmentation mask dimensions do not match input image."
    
if __name__ == "__main__":
    test_input_image_processor()
    print("test_input_image_processor passed.")
