from lunar_tools.cam import WebCam
from lunar_tools.movie import MovieSaver
import time
from datetime import datetime
import os
from tqdm import tqdm
import numpy as np
import pathlib
import argparse


def parse_args():
    """Parse command line arguments."""
    desc = "Record video from webcam."
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument("--prefix", type=str, default="test", help="Prefix for the output filename (default: test)")
    parser.add_argument("--duration", type=float, default=10.0, help="Recording duration in seconds (default: 10.0)")
    parser.add_argument("--fps", type=int, default=30, help="Target frames per second (default: 30)")
    parser.add_argument("--crf", type=int, default=0, help="Video quality (0=lossless, 51=worst, default: 0)")
    parser.add_argument("--countdown", type=int, default=5, help="Countdown before recording starts (default: 5)")
    parser.add_argument("--raw", action="store_true", help="Use raw pixel format (yuv444p instead of yuv420p)")
    return parser.parse_args()


def get_project_root():
    """Get absolute path to project root directory."""
    # This file is in experiments/noise_study/acquire_test_videos.py
    # So we need to go up two levels to get to the project root
    current_file = pathlib.Path(__file__).resolve()
    return str(current_file.parent.parent.parent)


def countdown(seconds):
    """Display a countdown timer."""
    for i in range(seconds, 0, -1):
        print(f"\rStarting recording in {i} seconds...", end="", flush=True)
        time.sleep(1)
    print("\rRecording started!            ")


def init_camera(max_retries=3, wait_time=2):
    """Initialize camera with retries."""
    for attempt in range(max_retries):
        try:
            attempt_num = attempt + 1
            msg = "Attempting to initialize camera " f"(attempt {attempt_num}/{max_retries})"
            print(msg)
            # Disable any frame processing
            cam = WebCam(
                cam_id=0,
                do_digital_exposure_accumulation=False,
            )
            # Disable color channel flipping
            cam.shift_colors = False
            cam.do_mirror = False

            # Wait for camera to stabilize
            time.sleep(wait_time)

            # Test if camera is working by getting a frame
            test_frame = cam.get_img()
            if test_frame is not None:
                print("Camera initialized successfully")
                print(f"Actual frame size: {test_frame.shape}")
                return cam, test_frame.shape

            print("Camera returned None frame, retrying...")
            cam.release()

        except Exception as e:
            print(f"Error initializing camera: {e}")
            if attempt < max_retries - 1:
                print("Retrying...")
                time.sleep(wait_time)

    raise RuntimeError("Failed to initialize camera after multiple attempts")


def main():
    # Parse command line arguments
    args = parse_args()

    # Create output directory if it doesn't exist
    project_root = get_project_root()
    video_dirs = ["experiments", "noise_study", "test_videos"]
    output_dir = os.path.join(project_root, *video_dirs)
    os.makedirs(output_dir, exist_ok=True)

    # Initialize webcam with retries and get actual frame size
    cam, frame_shape = init_camera()

    # Generate filename with current datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"{args.prefix}_{timestamp}.mp4")

    # Set up movie saver with lossless settings
    movie = MovieSaver(
        fp_out=output_file,
        fps=args.fps,
        shape_hw=frame_shape,
        crf=args.crf,  # 0 for lossless
        codec="libx264",  # x264 has good lossless support
        preset="ultrafast",  # Minimize processing
        pix_fmt="yuv444p" if args.raw else "yuv420p",  # Full chroma sampling if raw
        silent_ffmpeg=False,
    )

    try:
        # Countdown
        countdown(args.countdown)

        # Record for specified duration
        start_time = time.time()
        frames_written = 0
        expected_frames = int(args.duration * args.fps)
        last_frame = None
        frame_times = []

        # Initialize progress bar
        pbar = tqdm(total=expected_frames, desc="Recording", unit="frames")

        while time.time() - start_time < args.duration:
            frame = cam.get_img()
            if frame is not None:
                # Skip if frame is identical to last frame
                is_duplicate = last_frame is not None and np.array_equal(frame, last_frame)
                if is_duplicate:
                    continue

                # Store frame time
                frame_times.append(time.time() - start_time)

                # Write frame (MovieSaver expects RGB format)
                movie.write_frame(frame)
                frames_written += 1
                pbar.update(1)

                # Store current frame for comparison
                last_frame = frame.copy()

        pbar.close()

        # Calculate frame timing statistics
        frame_intervals = np.diff(frame_times)
        avg_interval = np.mean(frame_intervals)
        std_interval = np.std(frame_intervals)
        actual_fps = 1.0 / avg_interval if avg_interval > 0 else 0

        # Print recording statistics
        actual_duration = time.time() - start_time
        print("\nRecording complete!")
        print(f"Video saved to: {output_file}")
        print(f"Total frames captured: {frames_written}")
        print(f"Actual duration: {actual_duration:.1f} seconds")
        print(f"Average FPS: {actual_fps:.1f}")

        # Format frame interval string
        interval_ms = avg_interval * 1000
        std_ms = std_interval * 1000
        interval_str = f"Frame interval: {interval_ms:.1f}ms ± {std_ms:.1f}ms"
        print(interval_str)

    except KeyboardInterrupt:
        print("\nRecording interrupted!")

    finally:
        # Clean up
        movie.finalize()  # Important: this properly closes the video file
        cam.release()


if __name__ == "__main__":
    main()
