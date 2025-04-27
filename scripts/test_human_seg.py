import lunar_tools as lt
from rtd.utils.segmentation_detection import HumanSeg
import numpy as np
from rtd.utils.misc_utils import get_repo_path
import torch
import cv2
import time

def main():
    # Initialize video reader
    video_file_path = get_repo_path("materials/videos/long_cut4.mp4")
    print(f"Reading video from: {video_file_path}")
    
    # Try reading with OpenCV first to verify the video
    cap = cv2.VideoCapture(video_file_path)
    if not cap.isOpened():
        print("Error: Could not open video file with OpenCV")
        return
        
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video properties: {width}x{height} @ {fps}fps, {total_frames} frames")
    cap.release()
    
    # Initialize movie reader
    movie_reader = lt.MovieReader(video_file_path)
    
    # Initialize renderer with video dimensions
    height_render = 1080
    width_render = 1920
    renderer = lt.Renderer(
        width=width_render,
        height=height_render,
        backend="pygame",
        do_fullscreen=False,
    )
    
    # Initialize human segmentation
    human_seg = HumanSeg(
        size=(576//4, 1024//4),  # Specific size instead of resizing factor
        device="cuda:0",
        apply_smoothing=True,
        gaussian_kernel_size=9,
        gaussian_sigma=3
    )
    
    # Initialize FPS tracker
    fps_tracker = lt.FPSTracker()
    
    while True:
        t_processing_start = time.time()
        
        # Get frame from video
        img_cam = movie_reader.get_next_frame()
        
        # If we reached the end of the video, reset
        if img_cam is None or (isinstance(img_cam, np.ndarray) and (img_cam.size == 0 or np.max(img_cam) == 0)):
            print("End of video reached, looping back to the beginning")
            movie_reader = lt.MovieReader(video_file_path)
            img_cam = movie_reader.get_next_frame()
            if img_cam is None or (isinstance(img_cam, np.ndarray) and (img_cam.size == 0 or np.max(img_cam) == 0)):
                print("Error: Could not read frame after resetting video reader")
                break
        
        # Convert BGR to RGB for processing
        if img_cam is not None and img_cam.size > 0:
            img_cam = img_cam[:, :, ::-1].copy()
            
            # Ensure image is float32
            img_cam = img_cam.astype(np.float32)
            
            # Get human segmentation mask
            fps_tracker.start_segment("Segmentation")
            human_seg_mask = human_seg.get_mask(img_cam)
            
            # Apply mask to image
            fps_tracker.start_segment("Masking")
            img_masked = human_seg.apply_mask(img_cam)
            
            # Convert back to uint8 for rendering
            img_masked = img_masked.astype(np.uint8)
            
            # Render the result
            fps_tracker.start_segment("Rendering")
            renderer.render(img_masked)
            
            # Update and display FPS
            fps_tracker.print_fps()

if __name__ == "__main__":
    main() 