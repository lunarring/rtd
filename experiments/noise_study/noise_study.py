from lunar_tools.cam import WebCam
import sys
import time


def dynamic_print(text):
    """Print text that updates in place."""
    sys.stdout.write("\r" + text)
    sys.stdout.flush()


def main():
    # Initialize webcam
    cam = WebCam(cam_id=0, fixed_fps=15)

    try:
        while True:
            # Get current FPS
            current_fps = cam.get_fps()

            # Create status message
            status = f"Camera Framerate: {current_fps:.1f} FPS"
            dynamic_print(status)

            # Small sleep to prevent excessive CPU usage
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopping camera monitoring...")
        cam.release()


if __name__ == "__main__":
    main()
