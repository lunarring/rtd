import lunar_tools as lt
import cv2
import numpy as np
import time

def test_latency():
    # Initialize webcam
    shape_hw_cam = (480, 640)
    cam = lt.WebCam(shape_hw=shape_hw_cam)
    
    # Create a window to display the test pattern
    window_name = "Latency Test"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)
    
    # Start with black screen
    black_screen = np.zeros((600, 800, 3), dtype=np.uint8)
    white_screen = np.ones((600, 800, 3), dtype=np.uint8) * 255
    
    print("Point your camera at the screen and press any key when ready...")
    cv2.imshow(window_name, black_screen)
    cv2.waitKey(0)
    
    # Run multiple tests for better accuracy
    latencies = []
    for test in range(15):
        print(f"Running test {test+1}/5...")
        
        # Display black screen for a moment
        cv2.imshow(window_name, black_screen)
        cv2.waitKey(500)
        
        # Get initial frame to establish baseline
        img_cam = cam.get_img()
        initial_brightness = np.mean(img_cam)
        
        # Switch to white screen and record the time
        start_time = time.time()
        cv2.imshow(window_name, white_screen)
        cv2.waitKey(1)  # Update the display
        
        # Threshold for detecting significant brightness change (adjust as needed)
        brightness_threshold = initial_brightness * 1.1
        
        # Keep checking the camera until we detect the change
        while True:
            img_cam = cam.get_img()
            current_brightness = np.mean(img_cam)

            # print(f"current_brightness {current_brightness} initial_brightness {initial_brightness} brightness_threshold {brightness_threshold}")
            
            if current_brightness > brightness_threshold:
                end_time = time.time()
                latency = (end_time - start_time) * 1000  # Convert to milliseconds
                latencies.append(latency)
                print(f"Detected change! Latency: {latency:.2f} ms")
                break
            
            # Safety timeout (5 seconds)
            if time.time() - start_time > 5:
                print("Timeout - no change detected. Check camera positioning.")
                break
    
    # Calculate and display average latency
    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        print(f"\nAverage latency: {avg_latency:.2f} ms")
        print(f"Min latency: {min(latencies):.2f} ms")
        print(f"Max latency: {max(latencies):.2f} ms")
    
    # Clean up
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    test_latency()