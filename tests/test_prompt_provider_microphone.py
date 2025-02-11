import unittest
from src.rtd.utils.prompt_provider import PromptProviderMicrophone

class FakeAudioRecorder:
    def __init__(self):
        self.is_recording = False

class FakeSpeechDetector:
    def __init__(self, prompt="Test Prompt"):
        self.audio_recorder = FakeAudioRecorder()
        self.test_prompt = prompt

    def start_recording(self):
        self.audio_recorder.is_recording = True

    def stop_recording(self):
        self.audio_recorder.is_recording = False
        return self.test_prompt

class TestPromptProviderMicrophone(unittest.TestCase):
    def test_handle_mic_button(self):
        ppm = PromptProviderMicrophone()
        fake_sd = FakeSpeechDetector()
        ppm.speech_detector = fake_sd

        # Simulate mic press: should start recording.
        ppm.handle_mic_button(True)
        self.assertTrue(fake_sd.audio_recorder.is_recording, "Recording should start on mic press")

        # Simulate mic release: should stop recording and update the prompt.
        ppm.handle_mic_button(False)
        self.assertFalse(fake_sd.audio_recorder.is_recording, "Recording should stop on mic release")
        self.assertEqual(ppm.get_current_prompt(), "test prompt", "Current prompt should be updated to lowercased value")

if __name__ == "__main__":
    unittest.main()
