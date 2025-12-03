#!/usr/bin/env python3
"""Ultra-condensed paDC daemon - single file, dual-engine (GPU + OpenAI) buffer-based transcription"""

import os
import sys
import signal
import time
import threading
import queue
import subprocess
from pathlib import Path
from enum import Enum
from collections import deque
from datetime import datetime
import numpy as np
import pyperclip
from dotenv import load_dotenv
import sounddevice as sd
import wave
import io
import tempfile

load_dotenv()

# Paths
FIFO_PATH = Path("/tmp/padc.fifo")
PID_FILE = Path("/tmp/padc_daemon.pid")
LOG_FILE = Path("/tmp/padc_daemon.log")
STATUS_FILE = Path.home() / ".padc_status"

# Get project root (where the daemon script is located)
PROJECT_ROOT = Path(__file__).parent.parent
TRANSCRIPTION_LOG = PROJECT_ROOT / "transcriptions.log"
DEBUG_AUDIO_DIR = PROJECT_ROOT / "debug_audio"


class State(Enum):
    IDLE = "idle"
    RECORDING = "recording"


class RecordingMode(Enum):
    NORMAL = "normal"  # Only used for cancel
    INSERT = "insert"  # Paste with Shift+Insert
    INSERT_CONTINUE = "insert_continue"  # Paste with Shift+Insert, then auto-restart
    CLAUDE_SEND = "claude_send"  # Send to Claude via claude-send-active script


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def normalize_audio_buffer(audio_buffer: np.ndarray, target_level: float = 0.7) -> tuple[np.ndarray, dict]:
    """Normalize audio buffer to target level with intelligent gain control

    Args:
        audio_buffer: numpy array of audio samples (float32, -1.0 to 1.0)
        target_level: target peak level (0.0 to 1.0), 0.0 disables normalization

    Returns:
        tuple: (normalized_audio, info_dict) where info_dict contains normalization stats
    """
    if audio_buffer.size == 0 or target_level <= 0.0:
        return audio_buffer, {'normalized': False}

    # Flatten if needed
    audio_flat = audio_buffer.flatten() if audio_buffer.ndim > 1 else audio_buffer

    # Calculate RMS (root mean square) and peak levels
    rms_level = np.sqrt(np.mean(audio_flat ** 2))
    peak_level = np.abs(audio_flat).max()

    # If audio is completely silent, don't normalize
    if peak_level < 1e-6:
        return audio_buffer, {
            'normalized': False,
            'reason': 'silent',
            'peak_before': 0.0,
            'rms_before': 0.0
        }

    # Calculate gain needed to reach target level
    # Use peak-based normalization to prevent clipping
    gain = target_level / peak_level

    # Apply safety limit: don't amplify more than 20dB (10x)
    # This prevents over-amplification of very quiet recordings
    max_gain = 10.0
    if gain > max_gain:
        gain = max_gain

    # Apply gain
    normalized = audio_flat * gain

    # Final safety check: hard clip at ±1.0 (should rarely trigger)
    clipped = np.clip(normalized, -1.0, 1.0)
    clipping_occurred = not np.array_equal(normalized, clipped)

    # Reshape to original shape if needed
    if audio_buffer.ndim > 1:
        clipped = clipped.reshape(audio_buffer.shape)

    return clipped, {
        'normalized': True,
        'gain_db': 20 * np.log10(gain) if gain > 0 else 0.0,
        'gain_linear': gain,
        'peak_before': float(peak_level),
        'peak_after': float(np.abs(clipped).max()),
        'rms_before': float(rms_level),
        'rms_after': float(np.sqrt(np.mean(clipped.flatten() ** 2))),
        'clipping_occurred': clipping_occurred,
        'target_level': target_level
    }


def save_audio_buffer_to_wav(audio_buffer: np.ndarray, output_path: Path, sample_rate: int = 16000):
    """Save numpy audio buffer to WAV file in a separate thread

    Args:
        audio_buffer: numpy array of audio samples (float32, -1.0 to 1.0)
        output_path: Path where WAV file should be saved
        sample_rate: Sample rate of the audio (default 16000 Hz)
    """
    def _save():
        try:
            # Ensure directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert float32 [-1.0, 1.0] to int16 [-32768, 32767]
            audio_int16 = (audio_buffer * 32767).astype(np.int16)

            # Flatten if needed
            if audio_int16.ndim > 1:
                audio_int16 = audio_int16.flatten()

            # Write WAV file
            with wave.open(str(output_path), 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 2 bytes for int16
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_int16.tobytes())

            print(f"[{time.strftime('%H:%M:%S')}] Debug: Saved audio buffer to {output_path}", flush=True)
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] Warning: Failed to save debug audio: {e}", flush=True)

    # Run in separate thread to avoid blocking transcription
    threading.Thread(target=_save, daemon=True).start()


# ============================================================================
# INLINE AUDIO RECORDER
# ============================================================================

class AudioRecorder:
    """Inline audio recorder - records audio to numpy buffer"""

    def __init__(self, sample_rate: int = 16000, channels: int = 1, buffer_seconds: float = 60.0):
        self.sample_rate = sample_rate
        self.channels = channels
        self.buffer_seconds = buffer_seconds
        self.recording = False
        self.audio_queue = queue.Queue()

        # We'll calculate maxlen dynamically after seeing first chunk
        # Start with a large estimate, will be adjusted on first append
        self.audio_data = deque()
        self.chunks_per_second = None
        self.thread = None

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"Audio callback status: {status}")
        if self.recording:
            self.audio_queue.put(indata.copy())

    def play_chime(self):
        """Play a short pleasant chime sound to indicate recording has started"""
        duration = 0.15  # 150ms chime
        frequency = 440  # A5 note
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        # Create a simple sine wave with envelope for a pleasant chime
        envelope = np.exp(-3 * t)  # Exponential decay
        chime = envelope * np.sin(2 * np.pi * frequency * t)
        # Add a subtle second harmonic for richness
        chime += 0.3 * envelope * np.sin(2 * np.pi * frequency * 2 * t)
        # Normalize to prevent clipping (reduced volume for softer chime)
        chime = chime * 0.2
        sd.play(chime, self.sample_rate)
        sd.wait()  # Wait for the chime to finish

    def play_cancel_sound(self):
        """Play a descending tone to indicate recording was cancelled"""
        duration = 0.2  # 200ms sound
        start_freq = 330  # E4 note
        end_freq = 220  # A3 note (descending)
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        # Create a descending frequency sweep
        freq_sweep = np.linspace(start_freq, end_freq, len(t))
        phase = 2 * np.pi * np.cumsum(freq_sweep) / self.sample_rate
        # Apply envelope
        envelope = np.exp(-2 * t)  # Slower decay than start chime
        cancel_sound = envelope * np.sin(phase)
        # Reduced volume
        cancel_sound = cancel_sound * 0.2
        sd.play(cancel_sound, self.sample_rate)
        sd.wait()  # Wait for the sound to finish

    def start(self, play_chime=True):
        self.recording = True
        self.audio_data.clear()
        self.thread = threading.Thread(target=self._record_thread)
        self.thread.start()
        if play_chime:
            # Play chime in a separate thread to not block
            threading.Thread(target=self.play_chime).start()

    def _record_thread(self):
        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            callback=self._audio_callback,
            dtype=np.float32,
        ):
            while self.recording:
                try:
                    data = self.audio_queue.get(timeout=0.1)

                    # On first chunk, calculate proper maxlen based on actual chunk size
                    if self.chunks_per_second is None:
                        frames_per_chunk = len(data)
                        self.chunks_per_second = self.sample_rate / frames_per_chunk
                        max_chunks = int(self.chunks_per_second * self.buffer_seconds)
                        # Recreate deque with proper maxlen
                        self.audio_data = deque(self.audio_data, maxlen=max_chunks)

                    self.audio_data.append(data)
                except queue.Empty:
                    continue

    def get_buffer_snapshot(self) -> np.ndarray:
        """Get a copy of the current buffer without stopping recording"""
        if not self.audio_data:
            return np.array([])

        # Create a copy of current buffer and clear for next recording
        buffer_copy = list(self.audio_data)
        self.audio_data.clear()

        return np.concatenate(buffer_copy, axis=0)

    def stop(self) -> np.ndarray:
        """Stop recording completely (only used on shutdown)"""
        self.recording = False
        if self.thread:
            self.thread.join()

        if not self.audio_data:
            return np.array([])

        return np.concatenate(self.audio_data, axis=0)


# ============================================================================
# INLINE GPU WHISPER MODEL
# ============================================================================

class GPUWhisperModel:
    """Inline GPU-only Whisper model wrapper with contextual transcription"""

    def __init__(self, model_size: str = "base", max_context_tokens: int = 200, silence_cutoff_seconds: float = 20.0, language: str = "en"):
        self.model_size = model_size
        self.model = None
        self.device = "cuda"
        self.compute_type = "int8"
        self.context_tokens = []  # Store token IDs directly for efficiency
        self.max_context_tokens = max_context_tokens
        self.silence_cutoff_seconds = silence_cutoff_seconds
        self.language = language
        self.tokenizer = None
        self._initialize_model()

    def _check_cuda_available(self) -> bool:
        """Check if CUDA is properly available"""
        try:
            import torch
            if torch.cuda.is_available():
                # Try to actually use CUDA to ensure libraries are loaded
                test_tensor = torch.zeros(1).cuda()
                del test_tensor
                return True
        except Exception:
            pass
        return False

    def _initialize_model(self):
        """Initialize GPU Whisper model and tokenizer - exits on failure"""
        from faster_whisper import WhisperModel

        # Check if CUDA is actually available and working
        if not self._check_cuda_available():
            print("ERROR: CUDA not available or failed to initialize")
            STATUS_FILE.write_text("#[bg=red]ERROR#[default]")
            sys.exit(1)

        try:
            # Try GPU with int8 first (more compatible)
            self.model = WhisperModel(
                self.model_size,
                device="cuda",
                compute_type="int8"
            )
            self.compute_type = "int8"
            self.tokenizer = self.model.hf_tokenizer
            print(f"[{time.strftime('%H:%M:%S')}] Initialized GPU Whisper model: {self.model_size} (int8)")
            return
        except Exception as e:
            print(f"Failed with int8 on GPU: {e}")

            # Try float16 if int8 fails
            try:
                self.model = WhisperModel(
                    self.model_size,
                    device="cuda",
                    compute_type="float16"
                )
                self.compute_type = "float16"
                self.tokenizer = self.model.hf_tokenizer
                print(f"[{time.strftime('%H:%M:%S')}] Initialized GPU Whisper model: {self.model_size} (float16)")
                return
            except Exception as e2:
                print(f"Failed with float16 on GPU: {e2}")

        # No fallback - exit on failure
        print("ERROR: Failed to initialize GPU Whisper model")
        STATUS_FILE.write_text("#[bg=red]ERROR#[default]")
        sys.exit(1)

    def transcribe_buffer(self, audio_buffer: np.ndarray) -> tuple[str, dict]:
        """Transcribe audio buffer directly with contextual awareness

        Returns:
            tuple: (transcription_text, info_dict) where info_dict contains timing and VAD info
        """
        if not self.model:
            raise RuntimeError("GPU Whisper model not initialized")

        if audio_buffer.size == 0:
            return "", {}

        try:
            # Flatten to 1D if needed (faster-whisper expects 1D float32 array at 16kHz)
            if audio_buffer.ndim > 1:
                audio_buffer = audio_buffer.flatten()

            # Calculate audio duration
            audio_duration = len(audio_buffer) / 16000  # 16kHz sample rate

            # Prepare context from previous transcriptions (token-limited)
            context_text = None
            context_tokens_count = len(self.context_tokens)

            if self.context_tokens:
                # Decode tokens directly (no re-encoding needed!)
                # context_tokens is a list of token IDs (integers)
                context_text = self.tokenizer.decode(self.context_tokens, skip_special_tokens=True)

            # Transcribe directly from buffer with context
            lang = None
            if self.language != "auto":
                lang = self.language

            segments, whisper_info = self.model.transcribe(
                audio_buffer,
                beam_size=5,
                language=lang,
                condition_on_previous_text=True,
                initial_prompt=context_text,
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=500
                )
            )

            # Collect transcription and update context with VAD timing info
            new_text_words = []
            segments_list = list(segments)  # Force evaluation of generator

            # Filter out old segments separated by large silence gaps
            filtered_segments = []
            filtered_reason = None
            if len(segments_list) > 1 and self.silence_cutoff_seconds > 0:
                # Analyze gaps between segments from newest to oldest
                # NOTE: We use start-to-start gaps because VAD incorrectly sets segment.end
                # to the start of the next segment (known faster-whisper issue)
                for i in range(len(segments_list) - 1, 0, -1):
                    current_seg = segments_list[i]
                    prev_seg = segments_list[i - 1]

                    # Calculate time gap between start of consecutive segments
                    gap = current_seg.start - prev_seg.start

                    if gap > self.silence_cutoff_seconds:
                        # Found a large gap - discard all segments before this point
                        filtered_segments = segments_list[:i]
                        segments_list = segments_list[i:]
                        filtered_reason = f"{gap:.1f}s gap between segments"
                        break

            # Build info dict for logging
            info = {
                'buffer_duration': audio_duration,
                'context_tokens_before': context_tokens_count,
                'has_speech': len(segments_list) > 0,
                'language': whisper_info.language,
                'language_probability': whisper_info.language_probability
            }

            # Add filtered segment info if any were filtered
            if filtered_segments:
                info['filtered_segments'] = {
                    'count': len(filtered_segments),
                    'time_range': (filtered_segments[0].start, filtered_segments[-1].end),
                    'reason': filtered_reason,
                    'segments': [{'start': s.start, 'end': s.end, 'text': s.text} for s in filtered_segments]
                }

            if segments_list:
                first_segment = segments_list[0]
                last_segment = segments_list[-1]

                info.update({
                    'speech_start': first_segment.start,
                    'speech_end': last_segment.end,
                    'trimmed_start': first_segment.start,
                    'trimmed_end': audio_duration - last_segment.end,
                    'segments': []
                })

                for segment in segments_list:
                    info['segments'].append({
                        'start': segment.start,
                        'end': segment.end,
                        'text': segment.text
                    })
                    words = segment.text.strip().split()
                    new_text_words.extend(words)

            # Update context with new tokens (efficient: only tokenize new text once!)
            transcription = " ".join(new_text_words)
            if transcription:
                # Tokenize only the new text
                # encode() returns an Encoding object, get the token IDs with .ids
                encoding = self.tokenizer.encode(transcription)
                new_tokens = encoding.ids if hasattr(encoding, 'ids') else encoding
                # Append new tokens to context
                self.context_tokens.extend(new_tokens)
                # Trim to max_context_tokens (keep most recent)
                if len(self.context_tokens) > self.max_context_tokens:
                    self.context_tokens = self.context_tokens[-self.max_context_tokens:]

            info['context_tokens_after'] = len(self.context_tokens)

            return transcription.strip(), info
        except Exception as e:
            print(f"GPU transcription failed: {e}")
            return "", {}

    def reset_context(self):
        """Clear context without restarting model"""
        self.context_tokens = []


# ============================================================================
# INLINE OPENAI WHISPER MODEL
# ============================================================================

class OpenAIWhisperModel:
    """OpenAI Whisper API wrapper with contextual transcription"""

    def __init__(self, model_name: str = "whisper-1", max_context_words: int = 200):
        self.model_name = model_name
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
        self.context_words = deque(maxlen=max_context_words)
        self.max_context_words = max_context_words

        if not self.api_key or self.api_key == "your_openai_api_key_here":
            print(f"[{time.strftime('%H:%M:%S')}] WARNING: No valid OpenAI API key found")
            print(f"[{time.strftime('%H:%M:%S')}] OpenAI transcription will not be available")
            self.available = False
        else:
            print(f"[{time.strftime('%H:%M:%S')}] OpenAI Whisper API initialized ({model_name})")
            self.available = True

    def transcribe_buffer(self, audio_buffer: np.ndarray, language: str = "fr") -> tuple[str, dict]:
        """Transcribe audio buffer using OpenAI API with contextual awareness

        Args:
            audio_buffer: numpy array of audio samples (float32, -1.0 to 1.0)
            language: ISO 639-1 language code (default: 'fr' for French)

        Returns:
            tuple: (transcription_text, info_dict) where info_dict contains API info
        """
        if not self.available:
            return "", {"error": "OpenAI API key not configured"}

        if audio_buffer.size == 0:
            return "", {}

        try:
            # Import here to avoid dependency if not using OpenAI
            from openai import OpenAI

            # Flatten to 1D if needed
            if audio_buffer.ndim > 1:
                audio_buffer = audio_buffer.flatten()

            # Calculate audio duration
            audio_duration = len(audio_buffer) / 16000  # 16kHz sample rate

            # Prepare context from previous transcriptions
            context_text = None
            context_words_count = len(self.context_words)

            if self.context_words:
                context_text = " ".join(self.context_words)

            # Convert numpy buffer to WAV file in memory
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 2 bytes for int16
                wav_file.setframerate(16000)
                # Convert float32 [-1.0, 1.0] to int16 [-32768, 32767]
                audio_int16 = (audio_buffer * 32767).astype(np.int16)
                wav_file.writeframes(audio_int16.tobytes())

            # Reset buffer position for reading
            wav_buffer.seek(0)
            wav_buffer.name = "audio.wav"  # OpenAI API requires a filename

            # Call OpenAI API
            client = OpenAI(api_key=self.api_key)
            start_time = time.time()

            # Build transcription parameters
            if language == "auto":
                language = ""

            transcribe_params = {
                "model": self.model_name,
                "file": wav_buffer,
                "language": language,
                "response_format": "text"
            }

            # Add context as prompt if available
            if context_text:
                transcribe_params["prompt"] = context_text

            transcription = client.audio.transcriptions.create(**transcribe_params)

            api_time = time.time() - start_time

            # Extract text (OpenAI returns plain text with response_format="text")
            text = transcription.strip() if isinstance(transcription, str) else ""

            # Update context with new words
            if text:
                new_words = text.strip().split()
                self.context_words.extend(new_words)

            # Build info dict
            info = {
                'buffer_duration': audio_duration,
                'context_words_before': context_words_count,
                'context_words_after': len(self.context_words),
                'api_time': api_time,
                'engine': 'openai',
                'language': language,
                'has_speech': bool(text)
            }

            return text, info

        except ImportError:
            error_msg = "OpenAI package not installed. Run: uv add openai"
            print(f"[{time.strftime('%H:%M:%S')}] ERROR: {error_msg}")
            return "", {"error": error_msg}
        except Exception as e:
            error_msg = f"OpenAI API error: {e}"
            print(f"[{time.strftime('%H:%M:%S')}] ERROR: {error_msg}")
            return "", {"error": error_msg}

    def reset_context(self):
        """Clear context without restarting model"""
        self.context_words.clear()


# ============================================================================
# MAIN DAEMON
# ============================================================================

class PaDCDaemon:
    def __init__(self):
        self.state = State.IDLE
        # Get buffer size from environment or default to 30 seconds
        buffer_seconds = float(os.environ.get("PADC_BUFFER_SECONDS", "30.0"))
        self.recorder = AudioRecorder(buffer_seconds=buffer_seconds)
        self.whisper_gpu = None
        self.whisper_openai = None
        self.running = True
        self.recording_mode = RecordingMode.NORMAL
        self.is_processing = False
        self.context_reset_timer = None
        self.transcription_engine = "gpu"  # Default to GPU, can be "gpu" or "openai"
        # Get language from environment or default to English
        self.transcription_language = os.environ.get("PADC_LANGUAGE", "en")

        # Audio normalization configuration
        self.normalize_target = float(os.environ.get("PADC_NORMALIZE_AUDIO", "0.7"))
        if self.normalize_target > 0.0:
            print(f"[{time.strftime('%H:%M:%S')}] Audio normalization enabled (target: {self.normalize_target:.1%})")
        else:
            print(f"[{time.strftime('%H:%M:%S')}] Audio normalization disabled (using raw levels)")

        # Debug audio saving configuration
        self.debug_save_audio = os.environ.get("PADC_DEBUG_SAVE_AUDIO", "false").lower() == "true"
        if self.debug_save_audio:
            print(f"[{time.strftime('%H:%M:%S')}] Debug audio saving enabled -> {DEBUG_AUDIO_DIR}")

        # Silence gap filtering configuration
        self.silence_cutoff_seconds = float(os.environ.get("PADC_SILENCE_CUTOFF_SECONDS", "20.0"))
        if self.silence_cutoff_seconds > 0:
            print(f"[{time.strftime('%H:%M:%S')}] Silence gap filtering: {self.silence_cutoff_seconds:.1f}s threshold")
        else:
            print(f"[{time.strftime('%H:%M:%S')}] Silence gap filtering: disabled (keeping all segments)")

        # Initialize status file
        self._update_status_file()

        # Initialize both GPU and OpenAI Whisper models (heavy operation, done once at startup)
        self._init_whisper_models()

        # Signal handlers
        signal.signal(signal.SIGTERM, lambda s, f: self.shutdown())
        signal.signal(signal.SIGINT, lambda s, f: self.shutdown())

        self.start_recording()

    def _init_whisper_models(self):
        """Initialize both GPU and OpenAI Whisper models - runs once at startup"""
        # Initialize GPU model
        model_size = os.environ.get("PADC_MODEL", os.environ.get("WHISPER_MODEL", "base"))
        gpu_language = os.environ.get("PADC_LANGUAGE", "en")
        print(f"[{time.strftime('%H:%M:%S')}] Loading GPU Whisper model ({model_size}, language: {gpu_language})...")
        self.whisper_gpu = GPUWhisperModel(
            model_size=model_size,
            silence_cutoff_seconds=self.silence_cutoff_seconds,
            language=gpu_language
        )
        print(f"[{time.strftime('%H:%M:%S')}] GPU Whisper model loaded successfully")

        # Initialize OpenAI model
        openai_model = os.environ.get("PADC_OPENAI_MODEL", "whisper-1")
        print(f"[{time.strftime('%H:%M:%S')}] Initializing OpenAI Whisper API ({openai_model})...")
        self.whisper_openai = OpenAIWhisperModel(model_name=openai_model)
        if self.whisper_openai.available:
            print(f"[{time.strftime('%H:%M:%S')}] OpenAI Whisper API ready")

    def _auto_reset_context(self):
        """Auto-reset context after inactivity for both models"""
        if self.whisper_gpu:
            self.whisper_gpu.reset_context()
        if self.whisper_openai:
            self.whisper_openai.reset_context()
        print(f"\n[{time.strftime('%H:%M:%S')}]\n┌─ Context Reset\n│ Reason: 1 minute inactivity\n└─ Both GPU and OpenAI contexts cleared", flush=True)

    def _start_context_reset_timer(self):
        """Start timer to auto-reset context after 1 minute of inactivity"""
        if self.context_reset_timer:
            self.context_reset_timer.cancel()
        self.context_reset_timer = threading.Timer(60.0, self._auto_reset_context)
        self.context_reset_timer.daemon = True
        self.context_reset_timer.start()

    def _cancel_context_reset_timer(self):
        """Cancel context reset timer"""
        if self.context_reset_timer:
            self.context_reset_timer.cancel()
            self.context_reset_timer = None

    def _update_status_file(self):
        """Update status file for tmux status bar"""
        try:
            if self.is_processing:
                status = "#[bg=yellow]process#[default]"
            elif self.state == State.RECORDING:
                status = "online"
            else:
                status = "standby"
            STATUS_FILE.write_text(status)
        except Exception:
            pass  # Silently fail if can't write status file

    def process_text(self, text: str) -> str:
        """Process and correct commonly misheard words from Whisper"""
        if not text:
            return text

        # Word replacement mappings for common Whisper mistakes
        replacements = {
            "cloud": "Claude",
            "Cloud": "Claude",
        }

        # Apply replacements (word boundary aware)
        processed = text
        for wrong, correct in replacements.items():
            processed = processed.replace(wrong, correct)

        return processed

    def _log_transcription(self, text: str):
        """Log transcription to file with date and time"""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(TRANSCRIPTION_LOG, "a", encoding="utf-8") as f:
                f.write(f"{timestamp} {text}\n")
        except Exception as e:
            # Silently fail if can't write to log
            pass

    def start_recording(self):
        """Start recording audio"""
        if self.state == State.RECORDING:
            return "already_recording"

        # Cancel context reset timer when starting to record
        self._cancel_context_reset_timer()

        self.state = State.RECORDING
        self._update_status_file()
        self.recorder.start(play_chime=False)
        print(f"[{time.strftime('%H:%M:%S')}] Recording started... ", end="", flush=True)
        return "recording_started"

    def stop_recording(self):
        """Snapshot buffer and transcribe (recording continues)"""
        if self.state != State.RECORDING:
            return "not_recording"

        # Get snapshot of current buffer (clears buffer, keeps recording)
        audio_buffer = self.recorder.get_buffer_snapshot()

        self.is_processing = True
        self._update_status_file()
        processing_start_time = time.time()

        # Process in background with a copy of the buffer
        threading.Thread(
            target=self._transcribe,
            args=(audio_buffer, processing_start_time, self.recording_mode),
            daemon=True
        ).start()

        # Reset mode to normal after triggering transcription
        self.recording_mode = RecordingMode.NORMAL

        return "processing"

    def cancel_recording(self):
        """Drop audio buffer and reset context while continuing to record"""
        if self.state != State.RECORDING:
            return "not_recording"

        # Clear the audio buffer without stopping recording
        self.recorder.audio_data.clear()

        # Reset context immediately for both models
        if self.whisper_gpu:
            self.whisper_gpu.reset_context()
        if self.whisper_openai:
            self.whisper_openai.reset_context()

        # Reset mode to normal
        self.recording_mode = RecordingMode.NORMAL

        print(f"[{time.strftime('%H:%M:%S')}] Buffer cleared, context reset (still recording)", flush=True)
        return "buffer_cleared"

    def _transcribe(self, audio_buffer, processing_start_time, recording_mode):
        """Transcribe audio in background thread - direct buffer transcription"""
        try:
            if audio_buffer is None or audio_buffer.size == 0:
                print("... no audio captured", flush=True)
                return

            # Apply audio normalization if enabled
            norm_info = {}
            if self.normalize_target > 0.0:
                audio_buffer, norm_info = normalize_audio_buffer(audio_buffer, self.normalize_target)

            # Debug: Save audio buffer to WAV file if enabled (after normalization)
            if self.debug_save_audio:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                debug_wav_path = DEBUG_AUDIO_DIR / f"buffer_{timestamp}.wav"
                save_audio_buffer_to_wav(audio_buffer, debug_wav_path, sample_rate=16000)

            # Select transcription engine and transcribe
            if self.transcription_engine == "openai":
                if not self.whisper_openai or not self.whisper_openai.available:
                    print(f"\n[{time.strftime('%H:%M:%S')}] ERROR: OpenAI API not available, falling back to GPU", flush=True)
                    text, info = self.whisper_gpu.transcribe_buffer(audio_buffer)
                    info['engine'] = 'gpu_fallback'
                else:
                    text, info = self.whisper_openai.transcribe_buffer(audio_buffer, language=self.transcription_language)
            else:
                # Default to GPU
                text, info = self.whisper_gpu.transcribe_buffer(audio_buffer)
                info['engine'] = 'gpu'

            # Process text to fix common Whisper mistakes
            text = self.process_text(text)

            total_time = time.time() - processing_start_time

            # Build comprehensive log output
            if text:
                # Log transcription to file
                self._log_transcription(text)

                # Display consolidated transcription info
                timestamp = time.strftime('%H:%M:%S')
                log_lines = []
                log_lines.append(f"\n[{timestamp}]")

                # Show engine used
                engine = info.get('engine', 'gpu')
                engine_label = engine.upper()
                if engine == 'openai':
                    engine_label = f"OpenAI ({info.get('language', 'auto')})"
                elif engine == 'gpu_fallback':
                    engine_label = "GPU (fallback)"

                log_lines.append(f"┌─ Transcription [{engine_label}] ({total_time:.2f}s)")
                log_lines.append(f"│ Buffer: {info.get('buffer_duration', 0):.2f}s")

                # Show normalization info if applied
                if norm_info.get('normalized'):
                    gain_db = norm_info.get('gain_db', 0)
                    peak_before = norm_info.get('peak_before', 0)
                    peak_after = norm_info.get('peak_after', 0)
                    clipping = " [CLIPPED]" if norm_info.get('clipping_occurred') else ""
                    log_lines.append(f"│ Gain: {gain_db:+.1f}dB (peak: {peak_before:.1%} → {peak_after:.1%}){clipping}")

                # Show engine-specific info
                if engine == 'openai':
                    log_lines.append(f"│ API time: {info.get('api_time', 0):.2f}s")
                    log_lines.append(f"│ Context: {info.get('context_words_before', 0)} → {info.get('context_words_after', 0)} words")
                else:
                    if info.get('has_speech'):
                        log_lines.append(f"│ Speech: {info.get('speech_start', 0):.2f}s → {info.get('speech_end', 0):.2f}s")
                        log_lines.append(f"│ VAD trimmed: {info.get('trimmed_start', 0):.2f}s (start), {info.get('trimmed_end', 0):.2f}s (end)")
                        log_lines.append(f"│ Language: {info.get('language', '')} ({info.get('language_probability', 0)*100:.2f}%)")
                    log_lines.append(f"│ Context: {info.get('context_tokens_before', 0)} → {info.get('context_tokens_after', 0)} tokens")

                # Show filtered segments if any were discarded
                if info.get('filtered_segments'):
                    filtered = info['filtered_segments']
                    time_range = filtered['time_range']
                    log_lines.append(f"│ ⚠ Filtered: {filtered['count']} segments ({time_range[0]:.2f}s-{time_range[1]:.2f}s) - {filtered['reason']}")

                log_lines.append(f"│")

                # Show segments if available
                if info.get('segments'):
                    for i, seg in enumerate(info['segments'], 1):
                        log_lines.append(f"│ [{seg['start']:.2f}s-{seg['end']:.2f}s]{seg['text']}")

                log_lines.append(f"│")
                log_lines.append(f"│ Result: {text}")

                # Handle clipboard and insert modes
                if recording_mode == RecordingMode.CLAUDE_SEND:
                    # CLAUDE_SEND mode: don't touch clipboard, just send to script
                    insert_method = self._insert_with_claude_send(text)
                    log_lines.append(f"└─ ✓ Sent to Claude ({insert_method})")
                else:
                    # INSERT modes: copy to clipboard and paste
                    clipcontent = pyperclip.paste()
                    pyperclip.copy(text +  " ")

                    if recording_mode == RecordingMode.INSERT or recording_mode == RecordingMode.INSERT_CONTINUE:
                        insert_method = self._insert_with_xdotool(text)
                        time.sleep(0.5)  # Wait before restoring clipboard to prevent race conditions
                        pyperclip.copy(clipcontent)
                        log_lines.append(f"└─ ✓ Pasted ({insert_method})")
                    else:
                        log_lines.append(f"└─ ✓ Copied to clipboard")

                print("\n".join(log_lines), flush=True)
            else:
                timestamp = time.strftime('%H:%M:%S')
                if info.get('has_speech') is False:
                    print(f"\n[{timestamp}]\n┌─ Transcription ({total_time:.2f}s)\n│ Buffer: {info.get('buffer_duration', 0):.2f}s\n└─ No speech detected", flush=True)
                else:
                    print(f"\n[{timestamp}]\n... no speech detected ({total_time:.2f}s)", flush=True)

        except Exception as e:
            print(f"\n... transcription error: {e}", flush=True)
        finally:
            self.is_processing = False
            self._update_status_file()
            # Start context reset timer after every transcription
            # Will be cancelled if user starts recording again within 1 minute
            self._start_context_reset_timer()

    def _insert_with_xdotool(self, text):
        """Paste using tmux send-keys if marked pane exists, otherwise Shift+Insert"""
        try:
            # Check if there's a marked pane
            check_result = subprocess.run(
                ["bash", "-c", "tmux list-panes -a -F '#{pane_marked}' | grep -q 1"],
                capture_output=True
            )

            has_marked_pane = (check_result.returncode == 0)

            if has_marked_pane:
                # Use tmux send-keys to marked pane (add trailing space)
                subprocess.run(
                    ["tmux", "send-keys", "-t", "{marked}", text + " "],
                    check=True,
                    capture_output=True,
                    text=True
                )
                return "tmux"
            else:
                # Fallback to xdotool Shift+Insert
                subprocess.run(["xdotool", "key", "shift+Insert"], check=True)
                return "shift+Insert"

        except subprocess.CalledProcessError as e:
            print(f"[{time.strftime('%H:%M:%S')}] Insert error: {e}", flush=True)
            return "error"
        except FileNotFoundError as e:
            print(f"[{time.strftime('%H:%M:%S')}] Command not found: {e}", flush=True)
            return "error"

    def _insert_with_claude_send(self, text):
        """Send text to Claude using claude-send-active script"""
        try:
            result = subprocess.run(
                ["claude-send-active", text + " "],
                check=True,
                capture_output=True,
                text=True
            )
            return "claude-send-active"
        except subprocess.CalledProcessError as e:
            print(f"[{time.strftime('%H:%M:%S')}] claude-send-active error: {e}", flush=True)
            return "error"
        except FileNotFoundError:
            print(f"[{time.strftime('%H:%M:%S')}] claude-send-active not found in PATH", flush=True)
            return "error"

    def insert(self):
        """Stop recording and insert (no toggle, no auto-restart)"""
        if self.state != State.RECORDING:
            return "not_recording"

        # Play cancel sound in a separate thread to not block
        threading.Thread(target=self.recorder.play_cancel_sound).start()

        self.recording_mode = RecordingMode.INSERT
        return self.stop_recording()

    def claude_send(self):
        """Stop recording and send to Claude (no toggle, no auto-restart)"""
        if self.state != State.RECORDING:
            return "not_recording"

        # No audio notification for claude-send
        self.recording_mode = RecordingMode.CLAUDE_SEND
        return self.stop_recording()

    def toggle_insert(self):
        """Toggle recording with insert mode"""
        self.recording_mode = RecordingMode.INSERT
        return self.toggle()

    def toggle_insert_continue(self):
        """Toggle recording with insert-continue mode"""
        self.recording_mode = RecordingMode.INSERT_CONTINUE
        return self.toggle()

    def toggle(self):
        """Toggle recording state"""
        if self.state == State.IDLE:
            return self.start_recording()
        else:
            return self.stop_recording()

    def shutdown(self):
        """Shutdown daemon"""
        print(f"[{time.strftime('%H:%M:%S')}] Shutting down daemon...")
        self.running = False
        if self.state == State.RECORDING:
            self.recorder.stop()

        # Cancel context reset timer
        self._cancel_context_reset_timer()

        # Cleanup
        if FIFO_PATH.exists():
            os.unlink(FIFO_PATH)
        if PID_FILE.exists():
            os.unlink(PID_FILE)
        # Clear status file on shutdown
        try:
            STATUS_FILE.write_text("")
        except Exception:
            pass

        sys.exit(0)

    def reset_context(self, engine: str = "both"):
        """Reset transcription context

        Args:
            engine: Which engine to reset - "both", "gpu", or "openai"
        """
        if engine == "both" or engine == "gpu":
            if self.whisper_gpu:
                self.whisper_gpu.reset_context()

        if engine == "both" or engine == "openai":
            if self.whisper_openai:
                self.whisper_openai.reset_context()

        if engine == "both":
            return "context_reset_both"
        else:
            return f"context_reset_{engine}"

    def process_command(self, cmd):
        """Process commands - English (GPU) and French (OpenAI) variants"""
        cmd = cmd.strip().lower()

        # English commands (use GPU)
        if cmd == "insert":
            self.transcription_engine = "gpu"
            self.transcription_language = "en"
            return self.insert()
        elif cmd == "toggle-insert":
            self.transcription_engine = "gpu"
            self.transcription_language = "en"
            return self.toggle_insert()
        elif cmd == "toggle-insert-continue":
            self.transcription_engine = "gpu"
            self.transcription_language = "en"
            return self.toggle_insert_continue()
        elif cmd == "claude-send":
            self.transcription_engine = "gpu"
            self.transcription_language = "en"
            return self.claude_send()

        # French commands (use OpenAI)
        elif cmd == "insert-fr":
            self.transcription_engine = "openai"
            self.transcription_language = "fr"
            return self.insert()
        elif cmd == "toggle-insert-fr":
            self.transcription_engine = "openai"
            self.transcription_language = "fr"
            return self.toggle_insert()
        elif cmd == "toggle-insert-continue-fr":
            self.transcription_engine = "openai"
            self.transcription_language = "fr"
            return self.toggle_insert_continue()
        elif cmd == "claude-send-fr":
            self.transcription_engine = "openai"
            self.transcription_language = "fr"
            return self.claude_send()

        # Context reset commands
        elif cmd == "reset-context":
            return self.reset_context("both")
        elif cmd == "reset-context-gpu":
            return self.reset_context("gpu")
        elif cmd == "reset-context-openai":
            return self.reset_context("openai")

        # Other commands
        elif cmd == "cancel":
            return self.cancel_recording()
        elif cmd == "shutdown":
            self.shutdown()
        else:
            return f"unknown_command|{cmd}"

    def run(self):
        """Main daemon loop"""
        # Write PID
        PID_FILE.write_text(str(os.getpid()))

        # Create FIFO
        if FIFO_PATH.exists():
            os.unlink(FIFO_PATH)
        os.mkfifo(FIFO_PATH)
        # Set permissions to allow all users to write (666)
        os.chmod(FIFO_PATH, 0o666)

        print(
            f"[{time.strftime('%H:%M:%S')}] paDC daemon started (PID: {os.getpid()})",
            flush=True,
        )
        print(
            f"[{time.strftime('%H:%M:%S')}] Listening on {FIFO_PATH}",
            flush=True,
        )

        # Main loop - listen for commands
        while self.running:
            try:
                with open(FIFO_PATH, "r") as fifo:
                    for line in fifo:
                        if not self.running:
                            break

                        response = self.process_command(line)
                        if response and response not in ["recording_started", "processing", "recording_cancelled"]:
                            # Only print non-recording status messages since those are handled inline
                            print(
                                f"[{time.strftime('%H:%M:%S')}] {response}",
                                flush=True,
                            )

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"[{time.strftime('%H:%M:%S')}] Error: {e}")
                time.sleep(0.1)

        self.shutdown()


def main():
    """Entry point"""
    # Check if already running
    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text())
            os.kill(pid, 0)  # Check if process exists
            print(f"Daemon already running (PID: {pid})")
            sys.exit(1)
        except (ProcessLookupError, ValueError):
            PID_FILE.unlink()  # Clean up stale PID file

    # Daemonize unless --foreground flag
    if "--foreground" not in sys.argv:
        # First fork
        pid = os.fork()
        if pid > 0:
            print(f"Starting daemon (PID: {pid})")
            sys.exit(0)

        # New session
        os.setsid()

        # Second fork
        pid = os.fork()
        if pid > 0:
            sys.exit(0)

        # Redirect outputs with unbuffered mode (line buffering)
        sys.stdout = open(LOG_FILE, "a", buffering=1)  # Line buffered
        sys.stderr = sys.stdout

    daemon = PaDCDaemon()
    daemon.run()


if __name__ == "__main__":
    main()
