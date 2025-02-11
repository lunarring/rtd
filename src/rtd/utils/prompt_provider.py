#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
import time
import lunar_tools as lt

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
    
    def __init__(self, init_prompt: str = "Image of a cat"):
        super().__init__(init_prompt)
        self.speech_detector = lt.Speech2Text()
        # Add any microphone initialization here

    def get_current_prompt(self) -> str | bool:
        """
        Get the current prompt.
        
        Returns:
            str: The current prompt
            bool: False if no prompt is available
        """
        return self._current_prompt

    def start_recording(self):
        self.speech_detector.start_recording()

    def stop_recording(self):
        new_prompt = self.speech_detector.stop_recording()
        if new_prompt:
            self._current_prompt = new_prompt

    def handle_mic_button(self, mic_button_state: bool):
        if mic_button_state:
            if not self.speech_detector.audio_recorder.is_recording:
                self.start_recording()
        else:
            if self.speech_detector.audio_recorder.is_recording:
                try:
                    prompt_new = self.speech_detector.stop_recording()
                    prompt_new = prompt_new.strip().lower()
                    self._current_prompt = prompt_new
                except Exception as e:
                    print(f"Error stopping recording: {e}")


class PromptProviderDatabase(PromptProvider):
    """
    A prompt provider that gets prompts from a database.
    This is just an example implementation.
    """
    
    def __init__(self, database_url: str, init_prompt: str = "Image of a cat"):
        super().__init__(init_prompt)
        self._database_url = database_url
        # Add any database connection initialization here
        
    def get_current_prompt(self) -> str | bool:
        """
        Get the current prompt from the database if available.
        
        Returns:
            str: The next prompt from the database if one exists
            bool: False if no new prompts are available
        """
        # Add actual database query logic here
        # This is just a placeholder
        return self._current_prompt 