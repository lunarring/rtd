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
from signal import signal, SIGINT

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
        current_prompt = self.get_current_prompt()

        if current_prompt and current_prompt != self._last_prompt:
            self._last_prompt = current_prompt
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
        llm_system_prompt: str = (
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
            llm_system_prompt (str): System prompt for the LLM
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
            llm_system_prompt=llm_system_prompt,
            llm_max_tokens=max_tokens,
            llm_temperature=temperature,
            llm_min_words=min_words,
            llm_max_words=max_words,
            audio_file=audio_file,
            min_time_passed=min_time_passed,
        )

    def get_current_prompt(self) -> str | bool:
        """
        Get the current prompt from the LLM processed speech-to-text.

        Returns:
            str: The current LLM-processed prompt if available
            bool: False if no prompt is available
        """
        response = self.streamer.get_llm_response()
        if response:
            return response
        return False

    def new_prompt_available(self):
        new_prompt_available = self.streamer.is_new_llm_response()
        if new_prompt_available:
            self._last_prompt = self.streamer.get_llm_response()
        return new_prompt_available


if __name__ == "__main__":

    def signal_handler(sig, frame):
        print("\nExiting...")
        exit(0)

    signal(SIGINT, signal_handler)

    print("Simple PromptProviderSpeechToText Example")
    print("----------------------------------------")
    print("This example will listen to your speech and convert it to image prompts.")
    print("Speak clearly into your microphone.")
    print("Press Ctrl+C to exit.")

    # Create a speech-to-text prompt provider with default settings
    stt_provider = PromptProviderSpeechToText(
        init_prompt="A colorful sunset over mountains",
        llm_system_prompt=("Convert speech to vivid image prompts. Focus on visual elements."),
        temperature=0.8,
    )

    try:
        print(f"Initial prompt: {stt_provider._current_prompt}")
        print("Listening for speech... Speak now!")

        while True:
            # Check if there's a new prompt available
            if stt_provider.new_prompt_available():
                prompt = stt_provider.last_prompt
                print(f"\nNew prompt generated: {prompt}")

            # Small delay to prevent high CPU usage
            time.sleep(0.2)

    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        # Cleanup code could go here if needed
        print("Example completed.")
