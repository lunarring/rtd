import sounddevice as sd
import numpy as np
import threading
import signal
import time
from collections import deque

class AudioDetector:
    def __init__(self, sampling_rate=44100, normalize_to_baseline=True):
        self.sampling_rate = sampling_rate
        self.current_volume = 0
        self.max_volume_since_last_call = 0
        self.last_call_max_volume = 0
        self.lock = threading.Lock()
        self.running = True
        self.stream = None
        
        # New variable for baseline normalization
        self.normalize_to_baseline = normalize_to_baseline
        self.baseline_window = deque(maxlen=int(5 * sampling_rate / 100))  # Store 5 seconds of values (assuming 100ms sleep)
        self.baseline_mean = 0
        self.baseline_max = 1.0  # Initialize with a default value to avoid division by zero

        # Make thread daemon so it doesn't block program exit
        self.thread = threading.Thread(target=self._process_audio, daemon=True)
        self.thread.start()

        # Register signal handler for graceful shutdown
        self._original_sigint_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handle_sigint)

    def _audio_callback(self, indata, frames, time, status):
        if not self.running:
            return
        with self.lock:
            # Compute RMS (Root Mean Square) of the audio signal
            self.current_volume = np.linalg.norm(indata) * 10
            
            # If normalization is enabled, add to baseline window
            if self.normalize_to_baseline:
                self.baseline_window.append(self.current_volume)
                if len(self.baseline_window) > 0:
                    self.baseline_mean = np.mean(self.baseline_window)
                    self.baseline_max = max(np.max(self.baseline_window), 0.001)  # Avoid division by zero
            
            # Update the maximum volume since the last call
            if self.current_volume > self.max_volume_since_last_call:
                self.max_volume_since_last_call = self.current_volume

    def _process_audio(self):
        try:
            # Open the audio stream
            self.stream = sd.InputStream(callback=self._audio_callback, channels=1, samplerate=self.sampling_rate)
            with self.stream:
                while self.running:
                    # Use shorter sleep intervals to check running flag more frequently
                    sd.sleep(100)
        except Exception as e:
            print(f"Audio processing error: {e}")
        finally:
            # Ensure stream is closed
            if self.stream is not None and self.stream.active:
                self.stream.close()

    def _handle_sigint(self, sig, frame):
        # Restore original handler and stop gracefully
        signal.signal(signal.SIGINT, self._original_sigint_handler)
        self.stop()
        # Call original handler
        if self._original_sigint_handler:
            self._original_sigint_handler(sig, frame)

    def get_last_volume(self):
        with self.lock:
            # Return the max volume since the last call and reset
            max_volume = self.max_volume_since_last_call
            self.max_volume_since_last_call = 0
            
            # If normalization is enabled, subtract the baseline mean and normalize to 0..1 range
            if self.normalize_to_baseline and len(self.baseline_window) > 0:
                max_volume = max(0, max_volume - self.baseline_mean)
                # Normalize to 0..1 range using the max value from the same window
                max_volume = min(1.0, max_volume / self.baseline_max)
                max_volume = max_volume**2
                
            return max_volume

    def stop(self):
        self.running = False
        # Give the thread a short time to exit gracefully
        if self.thread.is_alive():
            self.thread.join(timeout=0.5)