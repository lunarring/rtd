#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
import time
import lunar_tools as lt
import random
import numpy as np
import logging
import threading
import argparse

# Import from RTD modules
from rtd.voice.speech_to_text_streaming import SpeechToTextStreamer


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("prompt_provider")


class PromptProvider(ABC):
    """
    Abstract base class for prompt providers.
    This class defines the interface that all prompt providers must implement.
    """

    def __init__(self, init_prompt: str = "Image of a cat"):
        """Initialize the prompt provider"""
        self._last_prompt_time = 0
        self._last_prompt = None
        self._current_prompt = init_prompt

    @property
    def last_prompt(self) -> str | None:
        """Get the last prompt that was returned"""
        return self._last_prompt

    @abstractmethod
    def get_current_prompt(self) -> str | bool:
        """
        Abstract method to get the current prompt from the provider.
        Must be implemented by subclasses.

        Returns:
            str: The current prompt if one is available
            bool: False if no prompt is available
        """
        pass

    def new_prompt_available(self) -> str | bool:
        """
        Get a new prompt if available and different from the last one.

        Returns:
            str: The new prompt if one is available and different
            bool: False if no new prompt is available or same as last prompt
        """

        if self._current_prompt and self._current_prompt != self._last_prompt:
            self._last_prompt = self._current_prompt
            self._last_prompt_time = time.time()
            return True
        else:
            return False


class PromptProviderMicrophone(PromptProvider):
    """
    A prompt provider that gets prompts from microphone input.
    This is just an example implementation.
    """

    def __init__(self, init_prompt: str = "Image of a cat", speech_detector=None):
        super().__init__(init_prompt)
        if speech_detector is None:
            self.speech_detector = lt.Speech2Text()
        else:
            self.speech_detector = speech_detector
        # Add any microphone initialization here

    def get_current_prompt(self) -> str | bool:
        """
        Get the current prompt.

        Returns:
            str: The current prompt
            bool: False if no prompt is available
        """
        return self.speech_detector.transcript

    def start_recording(self):
        self.speech_detector.start_recording()

    def stop_recording(self):
        new_prompt = self.speech_detector.stop_recording()
        if new_prompt:
            self._current_prompt = new_prompt

    def handle_unmute_button(self, mic_button_state: bool):
        return self.speech_detector.handle_unmute_button(mic_button_state)


class PromptProviderSpeechToText(PromptProvider):
    """
    A prompt provider that uses advanced speech-to-text streaming with LLM.
    This provider captures speech via microphone, transcribes it using OpenAI,
    and processes the transcription with an LLM to generate creative prompts.
    """

    def __init__(
        self,
        init_prompt: str = "Image of a cat",
        use_llm: bool = True,
        llm_model: str = "gpt-4o",
        system_prompt: str = (
            "Convert the user's speech into concise, vivid image generation prompts. "
            "Focus on visual elements and artistic style. Keep it under 20 words."
        ),
        max_tokens: int = 100,
        temperature: float = 0.7,
        min_words: int = 3,
        max_words: int = 30,
        audio_file: str = None,
        min_time_passed: float = 1.0,
        check_interval: float = 0.1,
    ):
        """
        Initialize the speech-to-text prompt provider.

        Args:
            init_prompt (str): Initial prompt to use
            use_llm (bool): Whether to use LLM processing on transcriptions
            llm_model (str): LLM model to use for processing
            system_prompt (str): System prompt for the LLM
            max_tokens (int): Maximum tokens for LLM responses
            temperature (float): LLM temperature setting
            min_words (int): Minimum words required for processing
            max_words (int): Maximum words to consider from transcripts
            audio_file (str): Optional audio file path instead of microphone
            min_time_passed (float): Minimum time between transcript checks
            check_interval (float): How often to check for updates (seconds)
        """
        super().__init__(init_prompt)

        # Set up the STT streamer with LLM capabilities
        self.streamer = SpeechToTextStreamer(
            use_llm=use_llm,
            llm_model=llm_model,
            llm_system_prompt=system_prompt,
            llm_max_tokens=max_tokens,
            llm_temperature=temperature,
            llm_min_words=min_words,
            llm_max_words=max_words,
            audio_file=audio_file,
            min_time_passed=min_time_passed,
        )

        # Additional settings
        self.check_interval = check_interval
        self.running = True
        self.lock = threading.Lock()

        # Start the monitor thread to check for new prompts
        self.monitor_thread = threading.Thread(target=self._monitor_for_new_prompts, daemon=True)
        self.monitor_thread.start()

        logger.info("Speech-to-Text prompt provider initialized")

    def get_current_prompt(self) -> str | bool:
        """
        Get the current prompt from the LLM-processed transcription.

        Returns:
            str: The current prompt
            bool: False if no prompt is available
        """
        return self._current_prompt

    def _monitor_for_new_prompts(self):
        """
        Background thread that monitors for new LLM responses
        and updates the current prompt when available.
        """
        while self.running:
            # Check for new LLM responses
            if self.streamer.is_new_llm_response():
                llm_response = self.streamer.get_llm_response()
                if llm_response:
                    with self.lock:
                        self._current_prompt = llm_response
                    logger.info(f"New prompt from LLM: {llm_response}")

            # If no LLM is used, check for direct transcriptions
            elif not self.streamer.use_llm and self.streamer.is_new_transcript():
                transcript = self.streamer.get_transcript()
                if transcript:
                    with self.lock:
                        self._current_prompt = transcript
                    logger.info(f"New prompt from transcript: {transcript}")

            time.sleep(self.check_interval)

    def stop(self):
        """Stop all processing threads and cleanup resources."""
        self.running = False
        self.streamer.stop_all()
        if self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2)
        logger.info("Speech-to-Text prompt provider stopped")


class PromptProviderTxtFile:
    def __init__(self, txt_file_path, mode="random"):
        """
        Initialize a prompt provider that reads prompts from a text file.

        Args:
            txt_file_path (str): Path to the text file containing prompts.
            mode (str): Mode for selecting prompts - "random" or "sequential".
                Default is "random".
        """
        self.txt_file_path = txt_file_path
        self.mode = mode
        self.prompts = []
        self.current_prompt = ""
        self.current_index = 0  # Track current position for sequential mode

        # Validate mode
        if mode not in ["random", "sequential"]:
            print(f"Warning: Unknown mode '{mode}'. Falling back to 'random' mode.")
            self.mode = "random"

        # Read prompts from file
        self.reload_prompts()

        # Initialize with a prompt
        self.get_next_prompt()

    def reload_prompts(self):
        """
        Reload prompts from the text file.
        """
        try:
            with open(self.txt_file_path, "r") as f:
                self.prompts = [line.strip() for line in f.readlines() if len(line.strip()) > 0]
            # For sequential mode, reset the index
            if self.mode == "sequential":
                self.current_index = 0
        except Exception as e:
            print(f"Error loading prompts from {self.txt_file_path}: {e}")
            self.prompts = ["A beautiful landscape"]  # Fallback prompt

    def get_next_prompt(self):
        """
        Get the next prompt according to the current mode.

        Returns:
            str: The next prompt.
        """
        if not self.prompts:
            return "A beautiful landscape"  # Fallback if no prompts are available

        if self.mode == "random":
            # Random mode: select a random prompt
            self.current_prompt = np.random.choice(self.prompts)
        else:  # sequential mode
            # Sequential mode: go through prompts in order
            self.current_prompt = self.prompts[self.current_index]
            # Move to next prompt, cycling back to the beginning if needed
            self.current_index = (self.current_index + 1) % len(self.prompts)

        return self.current_prompt

    def get_current_prompt(self):
        """
        Get the currently selected prompt.

        Returns:
            str: The current prompt.
        """
        return self.current_prompt

    def handle_prompt_cycling_button(self, button_pressed):
        """
        Handle the button press to cycle to the next prompt.

        Args:
            button_pressed (bool): Whether the button to cycle prompts was pressed.

        Returns:
            bool: True if a new prompt was selected, False otherwise.
        """
        if button_pressed:
            self.get_next_prompt()
            return True
        return False


if __name__ == "__main__":
    import time

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Speech-to-Text Prompt Provider Example")
    parser.add_argument("--demo", action="store_true", help="Run in demo mode with simulated speech")
    parser.add_argument("--model", type=str, default="gpt-4o", help="LLM model to use")
    args = parser.parse_args()

    print("Starting Speech-to-Text Prompt Provider Example")

    if args.demo:
        # Demo mode with simulated speech transcripts
        print("Running in DEMO MODE with simulated speech transcripts")

        # Create a simple class to simulate speech transcripts
        class SimulatedSpeechToTextStreamer:
            def __init__(self, *args, **kwargs):
                self.demo_transcripts = [
                    "a sunset over the ocean with silhouettes of palm trees",
                    "a futuristic city with flying cars and neon lights",
                    "a peaceful mountain landscape with a cabin and a lake",
                    "an ancient forest with mystical creatures and glowing plants",
                    "a busy marketplace in a medieval town",
                ]
                self.demo_responses = [
                    "Silhouettes of palm trees against vibrant sunset over infinite ocean",
                    "Neon-lit metropolis with sleek hovering vehicles cutting through skyscrapers",
                    "Serene log cabin beside crystal mountain lake reflecting snow-capped peaks",
                    "Ethereal woodland where luminous flora illuminates mythical beings",
                    "Medieval marketplace bustling with merchants, flags and ancient architecture",
                ]
                self.current_index = 0
                self.new_response = False
                self.use_llm = True

            def is_new_transcript(self, *args, **kwargs):
                # Only return True once
                return False

            def is_new_llm_response(self):
                # Alternate between True and False to simulate new responses
                if self.current_index < len(self.demo_responses) and not self.new_response:
                    self.new_response = True
                    return True
                return False

            def get_transcript(self, *args, **kwargs):
                return self.demo_transcripts[self.current_index]

            def get_llm_response(self):
                response = self.demo_responses[self.current_index]
                self.new_response = False
                self.current_index = (self.current_index + 1) % len(self.demo_responses)
                return response

            def stop_all(self):
                print("Stopping simulated streamer...")

        # Create a demo version that uses the simulation
        class DemoPromptProvider(PromptProviderSpeechToText):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                # Replace the streamer with our simulated one
                self.streamer = SimulatedSpeechToTextStreamer()

            def _monitor_for_new_prompts(self):
                """Override to add delays between simulated responses"""
                while self.running:
                    # Check for new LLM responses
                    if self.streamer.is_new_llm_response():
                        llm_response = self.streamer.get_llm_response()
                        if llm_response:
                            with self.lock:
                                self._current_prompt = llm_response
                            logger.info(f"New prompt from LLM: {llm_response}")
                            # Simulate "thinking" for user demo
                            print(f"\n[Demo] Simulated transcript: {self.streamer.get_transcript()}")
                            print(f"[Demo] Processing with LLM...", end="", flush=True)
                            time.sleep(1)
                            print(".", end="", flush=True)
                            time.sleep(1)
                            print(".", end="", flush=True)
                            time.sleep(1)
                            print(f" done!")
                    time.sleep(3)  # Longer delay for demo mode

        # Use the demo provider
        speech_provider = DemoPromptProvider(init_prompt="Welcome to the demo", llm_model=args.model)

    else:
        # Live mode with actual microphone
        print("Running with LIVE microphone input")
        print("Make sure your OpenAI API key is set in the environment")

        # Create the real speech-to-text prompt provider
        speech_provider = PromptProviderSpeechToText(
            init_prompt="Describe what you see around you",
            use_llm=True,
            llm_model=args.model,
            system_prompt=(
                "Transform the user's description into an artistic image prompt. "
                "Focus on visual elements, mood, lighting, and style. "
                "Keep it concise but vivid, under a maximum of 20 words."
            ),
            max_tokens=50,
            temperature=0.7,
        )

    try:
        print("\n" + "=" * 50)
        print("SPEECH-TO-TEXT PROMPT GENERATOR")
        print("=" * 50)

        if not args.demo:
            print("Speak into your microphone to generate prompts.")
        print("Press Ctrl+C to exit.")
        print("=" * 50 + "\n")

        # Main loop to check for new prompts
        prompt_count = 0

        while True:
            if speech_provider.new_prompt_available():
                prompt = speech_provider.last_prompt
                prompt_count += 1
                print(f"\n✨ NEW PROMPT #{prompt_count}: {prompt}")

                # Give visual separation for next prompt
                print("\n" + "-" * 50)

            # Wait a bit before checking again
            time.sleep(0.5)

            # For demo mode, exit after showing all examples
            if args.demo and prompt_count >= 5:
                print("\nDemo completed with all examples shown.")
                break

    except KeyboardInterrupt:
        print("\nStopping prompt provider...")
    finally:
        # Clean up resources
        speech_provider.stop()
        print("\nExample completed. Thank you for trying the Speech-to-Text Prompt Provider!")
