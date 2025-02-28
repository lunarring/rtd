#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
import time
import lunar_tools as lt
import random


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
            str: The new prompt if one is available and different from last prompt
            bool: False if no new prompt is available or if it's the same as last prompt
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


class PromptProviderTxtFile(PromptProvider):
    """
    A prompt provider that gets prompts from a text file.
    This is just an example implementation.
    """

    def __init__(self, file_path: str):
        super().__init__("picture of a cat")
        self._file_path = file_path
        self.load_prompts()
        self._current_prompt = self.list_prompts[0]

    def load_prompts(self):
        with open(self._file_path, "r", encoding="utf-8") as file:
            self.list_prompts = file.readlines()

    def handle_prompt_cycling_button(self, cycle_prompt_button_state: bool):
        """
        Get a random prompt from the list of prompts.
        """
        if cycle_prompt_button_state:
            random_index = random.randint(0, len(self.list_prompts) - 1)
            self._current_prompt = self.list_prompts[random_index]

    def get_current_prompt(self) -> str | bool:
        """
        Get the current prompt.
        """
        return self._current_prompt
