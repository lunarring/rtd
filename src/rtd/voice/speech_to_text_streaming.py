#!/usr/bin/env python3

"""
Real-time Speech-to-Text Streaming Example

This example shows how to use OpenAI's speech-to-text API to transcribe audio in real-time
with streaming transcription results.
"""

import asyncio
import numpy as np
import sounddevice as sd
import threading
import time
import logging
import wave
import os
from openai import AsyncOpenAI

# Configure logging first before any imports that might use it
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger("speech_to_text")

# For MP3 support
try:
    from pydub import AudioSegment

    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    logger.warning("pydub not installed. MP3 support is disabled. " "Install with: pip install pydub")

# Import from the agents package (assuming it's installed via pip)
from agents.voice import StreamedAudioInput
from agents.voice.models.openai_stt import OpenAISTTModel
from agents.voice.model import STTModelSettings

# Suppress errors from OpenAI agents - completely silence them
logging.getLogger("openai").setLevel(logging.CRITICAL)
logging.getLogger("openai.http_client").setLevel(logging.DEBUG)  # Set HTTP client logs to DEBUG level
logging.getLogger("agents").setLevel(logging.CRITICAL)

# Also suppress specific error messages from lower level loggers
for logger_name in ["openai.agents.trace", "openai.agents.client", "openai.agents.tools"]:
    logging.getLogger(logger_name).setLevel(logging.CRITICAL)
    logging.getLogger(logger_name).propagate = False

# Audio settings
SAMPLE_RATE = 24000
CHANNELS = 1
CHUNK_SIZE = int(SAMPLE_RATE * 0.1)  # 100ms chunks
DEFAULT_DEVICE = None  # Let sounddevice choose default


client = AsyncOpenAI()


class SpeechToTextStreamer:
    """
    Encapsulates the streaming transcription workflow including microphone input,
    transcription streaming, and graceful shutdown.
    """

    def __init__(
        self,
        use_llm=True,
        llm_model="gpt-4o-2024-08-06",
        llm_system_prompt="Describe the image in detail in vivid colors given what was said",
        llm_max_tokens=100,
        llm_temperature=0.7,
        llm_min_words=2,
        llm_max_words=33,
        audio_file=None,
        min_time_passed=2.0,
        eagerness="high",
    ):
        self.client = client
        self.audio_input = None
        self.stt_model = None
        self.session = None
        self.stream = None
        self.thread = None
        self.running = False
        self.loop = None
        # Collect all transcriptions with timestamps
        self.all_transcriptions = []
        # Flag to track new transcriptions
        self.new_transcript_available = False
        # Track last time transcripts were checked
        self.last_check_time = time.time()
        # Track word count of last checked transcript
        self.last_transcript_words = 0
        # Time between transcript checks
        self.min_time_passed = min_time_passed
        # Transcription eagerness setting
        self.eagerness = eagerness

        # Audio source setup
        # Path to audio file (if using file input)
        self.audio_file = audio_file
        # Use microphone if no file is provided
        self.use_mic = audio_file is None

        # LLM processing setup
        self.use_llm = use_llm
        self.llm_model = llm_model
        self.llm_system_prompt = llm_system_prompt
        self.llm_max_tokens = llm_max_tokens
        self.llm_temperature = llm_temperature
        self.llm_min_words = llm_min_words
        self.llm_max_words = llm_max_words
        self.llm_client = AsyncOpenAI()  # Separate client for LLM
        self.llm_thread = None
        self.llm_loop = None
        self.latest_transcript = ""
        self.latest_llm_response = ""
        self.new_llm_response = False

        # Start transcription in a separate thread automatically
        self.trans_thread()

        # Start LLM processing in a separate thread if enabled
        if self.use_llm:
            self.llm_thread_start()

    async def start(self):
        """Start the real-time speech-to-text transcription workflow."""
        logger.info("Starting real-time speech-to-text streaming...")
        if self.use_mic:
            logger.info("Speak into your microphone. Press Ctrl+C to exit.")
        else:
            logger.info(f"Using audio file: {self.audio_file}")

        self.running = True
        # Initialize audio input stream abstraction
        self.audio_input = StreamedAudioInput()

        # Initialize STT model
        self.stt_model = OpenAISTTModel(
            model="gpt-4o-transcribe",  # or "whisper-1"
            openai_client=self.client,
        )

        # Configure transcription settings
        settings = STTModelSettings(
            language="en",  # Force English language transcription
            # Configure turn detection (lower eagerness means wait longer)
            turn_detection={"type": "semantic_vad", "eagerness": self.eagerness},  # Options: high, medium, low
            # Add prompt for English transcription
            prompt="Transcribe the following audio in English language only",
            temperature=0,  # Lower temperature for more deterministic results
        )

        # Create a session for streaming transcription
        self.session = await self.stt_model.create_session(
            input=self.audio_input,
            settings=settings,
            trace_include_sensitive_data=False,
            trace_include_sensitive_audio_data=False,
        )

        # Start the transcription processing task
        transcription_task = asyncio.create_task(self._process_transcriptions())

        try:
            if self.use_mic:
                # Use microphone input
                await self._stream_from_microphone()
            else:
                # Use file input
                await self._stream_from_file()
        except KeyboardInterrupt:
            logger.info("Stopping due to user interrupt...")
        finally:
            await self.stop()
            logger.info("Session closed")
            # Cancel the transcription processing task if still running
            transcription_task.cancel()
            try:
                await transcription_task
            except asyncio.CancelledError:
                pass

    async def _stream_from_microphone(self):
        """Stream audio from the microphone to the transcription service."""
        # Start microphone stream with explicit device selection
        try:
            self.stream = sd.InputStream(
                channels=CHANNELS,
                samplerate=SAMPLE_RATE,
                dtype="int16",
                device=DEFAULT_DEVICE,  # Use system default device
                blocksize=CHUNK_SIZE,
                latency="low",  # Request low latency
            )
            self.stream.start()
            logger.info("Microphone stream started successfully")

            # Continuously stream audio to the transcription service
            while self.running:
                try:
                    # Read audio data from microphone
                    data, _ = self.stream.read(CHUNK_SIZE)
                    # Send audio data to the transcription session
                    await self.audio_input.add_audio(data)
                except Exception as e:
                    logger.error(f"Error reading from microphone: {e}")
                    # Brief pause to prevent tight loop in case of errors
                    await asyncio.sleep(0.1)
                    continue

                await asyncio.sleep(0.01)  # Small delay to prevent CPU hogging

        except Exception as e:
            logger.error(f"Failed to initialize microphone: {e}")
            self.running = False

    async def _stream_from_file(self):
        """Stream audio from a file to the transcription service."""
        if not self.audio_file:
            logger.error("No audio file specified")
            return

        try:
            # Check file extension
            file_ext = os.path.splitext(self.audio_file)[1].lower()

            # Handle MP3 files using pydub
            if file_ext == ".mp3":
                if not PYDUB_AVAILABLE:
                    logger.error("Cannot process MP3 files. " "Please install pydub: pip install pydub")
                    return

                logger.info(f"Loading MP3 file: {self.audio_file}")
                # Convert MP3 to WAV in memory
                audio = AudioSegment.from_mp3(self.audio_file)

                # Adjust channels if needed
                if audio.channels != CHANNELS:
                    logger.warning(f"Converting from {audio.channels} channels " f"to {CHANNELS} channel(s)")
                    audio = audio.set_channels(CHANNELS)

                # Adjust sample rate if needed
                if audio.frame_rate != SAMPLE_RATE:
                    logger.warning(f"Converting from {audio.frame_rate}Hz " f"to {SAMPLE_RATE}Hz")
                    audio = audio.set_frame_rate(SAMPLE_RATE)

                logger.info(f"Audio duration: {audio.duration_seconds:.2f}s")

                # Stream audio data in chunks
                chunk_size = int(SAMPLE_RATE * 0.1)  # 100ms chunks
                chunk_ms = 100  # 100ms chunks

                for i in range(0, len(audio), chunk_ms):
                    if not self.running:
                        break

                    # Get chunk and convert to numpy array
                    chunk = audio[i : i + chunk_ms]
                    # Convert to the right format for the transcription API
                    samples = np.array(chunk.get_array_of_samples(), dtype=np.int16)

                    # Send audio data to the transcription session
                    await self.audio_input.add_audio(samples)

                    # Simulate real-time playback by waiting
                    await asyncio.sleep(0.1)  # Wait 100ms between chunks

                logger.info("Finished streaming MP3 file")
                # Wait for a moment to let the transcriber finish processing
                await asyncio.sleep(2)
                return

            # Handle WAV files using built-in wave module
            elif file_ext == ".wav":
                # Open the wave file
                with wave.open(self.audio_file, "rb") as wf:
                    # Check if the file format is compatible
                    if wf.getnchannels() != CHANNELS:
                        logger.warning(f"File has {wf.getnchannels()} channels, " f"expected {CHANNELS}")

                    sample_rate = wf.getframerate()
                    if sample_rate != SAMPLE_RATE:
                        logger.warning(f"File sample rate is {sample_rate}Hz, " f"expected {SAMPLE_RATE}Hz")

                    logger.info(f"Audio file: {self.audio_file}, " f"duration: {wf.getnframes() / sample_rate:.2f}s")

                    # Read and stream the file data in chunks
                    chunk_size = int(sample_rate * 0.1)  # 100ms chunks
                    data = wf.readframes(chunk_size)

                    while data and self.running:
                        # Convert data to numpy array
                        audio_data = np.frombuffer(data, dtype=np.int16)
                        # Stream to transcription service
                        await self.audio_input.add_audio(audio_data)
                        # Read next chunk
                        data = wf.readframes(chunk_size)
                        # Simulate real-time playback by waiting
                        await asyncio.sleep(0.1)  # Wait 100ms between chunks

                    logger.info("Finished streaming WAV file")
                    # Wait for transcriber to finish processing
                    await asyncio.sleep(2)
                return

            # Unsupported file format
            else:
                logger.error(f"Unsupported audio format: {file_ext}. " f"Supported formats: .wav, .mp3")
                return

        except Exception as e:
            logger.error(f"Error streaming from file: {e}", exc_info=True)

    async def stop(self):
        """Gracefully shutdown the transcription session and clean up resources."""
        self.running = False
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
        if self.session is not None:
            await self.session.close()

    async def _process_transcriptions(self):
        """Process and display streaming transcription results."""
        try:
            async for transcription in self.session.transcribe_turns():
                # Each transcription represents a complete utterance
                timestamp = time.time()
                self.all_transcriptions.append({"text": transcription, "timestamp": timestamp})
                # Set flag when new transcript is available
                self.new_transcript_available = True

                # Directly update latest transcript for LLM processing if enabled
                # Get accumulated transcript for LLM instead of just the latest utterance
                if self.use_llm:
                    # Use get_transcript to get the full context (all words within time window)
                    full_transcript = self.get_transcript(delta_time=30, min_words=self.llm_min_words, max_words=self.llm_max_words)
                    if full_transcript:
                        self.latest_transcript = full_transcript

                logger.debug(f"Transcription: {transcription}")
        except asyncio.CancelledError:
            # Task was cancelled, exit gracefully
            pass
        except Exception as e:
            logger.error(f"Error in transcription: {e}", exc_info=True)

    def run_transcription(self):
        """Set up asyncio event loop and run the async transcription process."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.create_task(self.start())
        try:
            self.loop.run_forever()
        finally:
            pending = asyncio.all_tasks(self.loop)
            for task in pending:
                task.cancel()
            try:
                self.loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception as e:
                logger.error(f"Error during task cleanup: {e}")
                pass
            self.loop.close()

    def trans_thread(self):
        """Start the transcription process in a separate thread."""
        self.thread = threading.Thread(target=self.run_transcription, daemon=True)
        self.thread.start()

    def is_new_transcript(self, min_extra_words=1):
        """
        Check if a new transcript is available that meets criteria.

        Args:
            min_extra_words (int): Minimum additional words required (default 1)

        Returns:
            bool: True if new transcript meeting criteria is available
        """
        current_time = time.time()
        time_passed = current_time - self.last_check_time

        # Check if we have any transcriptions
        if not self.all_transcriptions:
            return False

        # Get total word count from all transcriptions
        current_word_count = sum(len(t["text"].split()) for t in self.all_transcriptions)
        extra_words = current_word_count - self.last_transcript_words

        # Check if criteria are met
        if self.new_transcript_available and time_passed >= self.min_time_passed and extra_words >= min_extra_words:
            self.new_transcript_available = False
            self.last_check_time = current_time
            self.last_transcript_words = current_word_count

            return True

        return False

    def get_transcript(self, delta_time=None, min_words=1, max_words=None):
        """
        Get merged transcriptions based on time window or word limits.

        Args:
            delta_time (float, optional): Time window in seconds
                If provided, time-based filtering is used
            min_words (int): Minimum number of words required (default 1)
            max_words (int, optional): Maximum number of words to return
                If provided, word-based limiting is used

        Returns:
            str: Merged transcription text or empty string if criteria not met
        """
        if not self.all_transcriptions:
            return ""

        # Sort all transcriptions in chronological order
        sorted_transcriptions = sorted(self.all_transcriptions, key=lambda x: x["timestamp"])

        # Apply time-based filtering if delta_time is specified
        if delta_time is not None:
            current_time = time.time()
            cutoff_time = current_time - delta_time
            sorted_transcriptions = [t for t in sorted_transcriptions if t["timestamp"] >= cutoff_time]

        # Extract all words in chronological order
        all_words = []
        for trans in sorted_transcriptions:
            all_words.extend(trans["text"].split())

        # Apply word-based limiting if max_words is specified
        if max_words is not None and len(all_words) > max_words:
            all_words = all_words[-max_words:]

        # Convert to string and check if meets minimum word count
        if len(all_words) >= min_words:
            return " ".join(all_words)

        return ""

    # LLM processing methods
    async def process_with_llm(self, text):
        """Process text with LLM and return response"""
        if not text or len(text.split()) < self.llm_min_words:
            return None

        try:
            response = await self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "system", "content": self.llm_system_prompt}, {"role": "user", "content": text}],
                max_tokens=self.llm_max_tokens,
                temperature=self.llm_temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error from LLM: {e}")
            return None

    async def llm_processing_loop(self):
        """Background loop to process transcripts with LLM"""
        logger.info("Starting LLM processing loop")
        last_processed_transcript = ""
        while self.running:
            if self.latest_transcript and self.latest_transcript != last_processed_transcript:
                transcript = self.latest_transcript
                # Save to avoid duplicate processing
                last_processed_transcript = transcript

                logger.info(f"LLM processing transcript: {transcript}")

                # Process with LLM
                response = await self.process_with_llm(transcript)
                if response:
                    self.latest_llm_response = response
                    self.new_llm_response = True
                    logger.info(f"LLM response: {response}")
                else:
                    logger.warning(f"LLM returned no response for: {transcript}")

            await asyncio.sleep(0.1)  # Short sleep to prevent CPU hogging

    def run_llm_loop(self):
        """Set up asyncio event loop for LLM processing"""
        self.llm_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.llm_loop)
        self.llm_loop.create_task(self.llm_processing_loop())

        try:
            self.llm_loop.run_forever()
        finally:
            pending = asyncio.all_tasks(self.llm_loop)
            for task in pending:
                task.cancel()
            try:
                self.llm_loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception as e:
                logger.error(f"Error during LLM task cleanup: {e}")
            self.llm_loop.close()

    def llm_thread_start(self):
        """Start the LLM processing in a separate thread"""
        self.llm_thread = threading.Thread(target=self.run_llm_loop, daemon=True)
        self.llm_thread.start()

    def is_new_llm_response(self):
        """Check if a new LLM response is available"""
        if self.new_llm_response:
            self.new_llm_response = False
            return True
        return False

    def get_llm_response(self):
        """Get the latest LLM response"""
        return self.latest_llm_response

    def get_latest_transcript(self):
        """Get the latest LLM response"""
        return self.latest_transcript 

    def stop_all(self):
        """Stop all processing (both STT and LLM)"""
        self.running = False

        # Stop STT processing
        if self.loop is not None:
            stop_future = asyncio.run_coroutine_threadsafe(self.stop(), self.loop)
            try:
                stop_future.result(timeout=5)
            except Exception as e:
                logger.error(f"Error during STT stop: {e}")
            self.loop.call_soon_threadsafe(self.loop.stop)

        # Stop LLM processing
        if self.use_llm and self.llm_loop is not None:
            self.llm_loop.call_soon_threadsafe(self.llm_loop.stop)

        # Join threads
        if self.thread is not None:
            self.thread.join(timeout=2)
        if self.use_llm and self.llm_thread is not None:
            self.llm_thread.join(timeout=2)


if __name__ == "__main__":
    print("Starting Speech-to-Text Streaming application")

    prompt = """Your task is to directly translate from poetry into visual descriptions. I give you couple examples and you can do it then for the next one. Here are rules: Your task is to produce a valid and new output. Keep the style and length of the examples in your output. Don't make it longer.

Input: The surface is not motionless. Output: Waves on the surface of endless blue ocean and flickering, silver light 
Input: movement originates from the tides Output: regular movement of waves depends on the (invisible) underwater tides 
Input: pulsations in the bloodstream Output: pulsating liquid flowing through a riverbed
Input: it ripples, reflected on the surface Output: ripples on the surface of magma
Input: First, there's an impulse, a vibration Output: impulse of kinetic energy turning into vibration of particles 
Input: reflection of the waves on sand Output: pattern on the sand resembles the shape of a wave
Input: dunes in slow-motion Output: grains of sand carried by the wind are forming dunes
Input: birds respond to magnetic signals Output: flock of birds, lines of flight
Input: reverberating , undulating shapes Output: undulating shapes in an empty space
Input: marked on the sand, on the water, in the air Output: transforming fractal  patterns, green
Input: inscribed into the bodies of the rocks Output: fossils and geological lines marked on the rocks
Input: The Metamorphosis of Time Output: transformations of a celestial mass happening across deep time
Input: on the still mirror of the lake Output: still lake hidden in a forest, glossy surface
Input: whispers of a breeze, water spirits murmuring their songs Output: breeze moving a surface of water
Input: break the stillness of the air Output: subtle frequencies disturbing the  air
Input: based on frequencies and undulations of the sound Output: sound undulating in a void, bright light
Input: the lullaby Output: silent landscape, starry sky
Input: the poem from the past appearing suddenly in your memory Output: blurry image of a glacier
Input: a turbulence Output: image of a gray sky, turbulence
Input: Wind is hitting my skin and I imagine its waves spreading into a kaleidoscope of light Output: Wind blows spreading into a kaleidoscope of light
Input: particles rolling on the surface Output:  l glass beads rolling on the surface of a petri dish
Input: atoms crashing, constant annihilations and re-productions Output:  elements colliding and dispersing into atoms 
Input: The Pattern of Movement, a ripple Output: trembling  pattern of movement in the kelp forest

The next message I send you will be an Input, and you directly continue after 'Output:'
"""

    # Instantiate the SpeechToTextStreamer with default settings
    streamer = SpeechToTextStreamer(
        llm_system_prompt=prompt,
    )

    try:
        # Main thread runs a synchronous heartbeat loop
        while True:
            # Check for new LLM responses
            if streamer.is_new_llm_response():
                llm_response = streamer.get_llm_response()
                print(f"LLM: {llm_response}")

            time.sleep(0.1)
    except KeyboardInterrupt:
        # On interruption, gracefully shutdown all processing
        print("Keyboard interrupt received, shutting down")
        streamer.stop_all()
    except Exception as e:
        print(f"Unexpected error: {e}")
        streamer.stop_all()
    finally:
        print("Application shutdown complete")
