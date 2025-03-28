#!/usr/bin/env python3
import argparse
import signal
import sys
import time
import numpy as np
from tqdm import tqdm
import cv2

from lunar_tools.cam import WebCam
from lunar_tools.movie import MovieSaver

def parse_args():
    parser = argparse.ArgumentParser(description='Record video from camera and save to file.')
    parser.add_argument('--output', '-o', type=str, default='recorded_video.mp4',
                      help='Output filename for the video (default: recorded_video.mp4)')
    parser.add_argument('--duration', '-d', type=float, default=10.0,
                      help='Duration of recording in seconds (default: 10.0)')
    parser.add_argument('--fps', type=int, default=24,
                      help='Frames per second for the output video (default: 24)')
    parser.add_argument('--cam-id', type=int, default=0,
                      help='Camera ID to use (default: 0)')
    parser.add_argument('--resolution', type=str, default='1024x576',
                      help='Resolution in format WIDTHxHEIGHT (default: 1024x576)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Parse resolution
    width, height = map(int, args.resolution.split('x'))
    # WebCam takes height, width
    shape_hw = (height, width)
    
    # Initialize camera
    print(f"Initializing camera {args.cam_id} with resolution {width}x{height}...")
    cam = WebCam(cam_id=args.cam_id, shape_hw=shape_hw)
    
    # Get a test frame to determine actual camera resolution
    test_frame = cam.get_img()
    if test_frame is None or test_frame.size == 0:
        print("Error: Could not get a valid frame from the camera.")
        cam.release()
        sys.exit(1)
    
    actual_height, actual_width = test_frame.shape[:2]
    print(f"Actual camera frame size: {actual_width}x{actual_height}")
    
    # Initialize movie saver with the actual frame dimensions
    # Pass shape_hw as a list instead of tuple to avoid the 'append' error
    print(f"Setting up video recording to {args.output} at {args.fps} fps...")
    movie_saver = MovieSaver(fp_out=args.output, fps=args.fps, shape_hw=[actual_height, actual_width])
    
    # Setup signal handling for graceful exit
    def signal_handler(sig, frame):
        print("\nInterrupt received, finalizing recording...")
        movie_saver.finalize()
        cam.release()
        print("Recording finalized successfully.")
        sys.exit(0)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Calculate total frames based on duration and fps
    total_frames = int(args.duration * args.fps)
    
    try:
        # Record video
        print(f"Starting recording for {args.duration} seconds ({total_frames} frames)...")
        
        # Get start time for FPS control
        start_time = time.time()
        frames_captured = 0
        
        # Use tqdm for progress bar
        for i in tqdm(range(total_frames)):
            # Get frame from camera
            frame = cam.get_img()
            
            # Write frame to movie
            if frame is not None and frame.size > 0:
                # Ensure frame has the expected dimensions
                if frame.shape[:2] != (actual_height, actual_width):
                    frame = cv2.resize(frame, (actual_width, actual_height))
                
                movie_saver.write_frame(frame)
                frames_captured += 1
            
            # Calculate actual elapsed time for this frame
            elapsed = time.time() - start_time
            # Calculate target time based on desired fps
            target_time = i / args.fps
            # Sleep if we're ahead of schedule
            if elapsed < target_time:
                time.sleep(target_time - elapsed)
        
        # Finalize movie
        print(f"Recording complete. Captured {frames_captured} frames in {time.time() - start_time:.2f} seconds.")
        movie_saver.finalize()
        cam.release()
        
    except Exception as e:
        print(f"Error during recording: {e}")
        # Ensure we finalize the movie in case of error
        movie_saver.finalize()
        cam.release()
        sys.exit(1)

if __name__ == "__main__":
    main()
