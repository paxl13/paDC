#!/usr/bin/env python3
"""Ultra-condensed paDC daemon - single file, GPU-only, buffer-based transcription"""

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

load_dotenv()

# Paths
FIFO_PATH = Path("/tmp/padc.fifo")
PID_FILE = Path("/tmp/padc_daemon.pid")
LOG_FILE = Path("/tmp/padc_daemon.log")
STATUS_FILE = Path.home() / ".padc_status"

# Get project root (where the daemon script is located)
PROJECT_ROOT = Path(__file__).parent.parent
TRANSCRIPTION_LOG = PROJECT_ROOT / "transcriptions.log"


class State(Enum):
    IDLE = "idle"
    RECORDING = "recording"


class RecordingMode(Enum):
    NORMAL = "normal"  # Only used for cancel
    INSERT = "insert"  # Paste with Shift+Insert
    INSERT_CONTINUE = "insert_continue"  # Paste with Shift+Insert, then auto-restart


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

    def __init__(self, model_size: str = "base", max_context_words: int = 200):
        self.model_size = model_size
        self.model = None
        self.device = "cuda"
        self.compute_type = "int8"
        self.context = []
        self.max_context_words = max_context_words
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
        """Initialize GPU Whisper model - exits on failure"""
        import logging
        from faster_whisper import WhisperModel

        # Enable debug logging for faster-whisper
        logging.basicConfig()
        logging.getLogger("faster_whisper").setLevel(logging.DEBUG)

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
            print(f"Initialized GPU Whisper model: {self.model_size} (int8)")
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
                print(f"Initialized GPU Whisper model: {self.model_size} (float16)")
                return
            except Exception as e2:
                print(f"Failed with float16 on GPU: {e2}")

        # No fallback - exit on failure
        print("ERROR: Failed to initialize GPU Whisper model")
        STATUS_FILE.write_text("#[bg=red]ERROR#[default]")
        sys.exit(1)

    def transcribe_buffer(self, audio_buffer: np.ndarray) -> str:
        """Transcribe audio buffer directly with contextual awareness"""
        if not self.model:
            raise RuntimeError("GPU Whisper model not initialized")

        if audio_buffer.size == 0:
            return ""

        try:
            # Flatten to 1D if needed (faster-whisper expects 1D float32 array at 16kHz)
            if audio_buffer.ndim > 1:
                audio_buffer = audio_buffer.flatten()

            # Calculate audio duration
            audio_duration = len(audio_buffer) / 16000  # 16kHz sample rate
            print(f"[Audio buffer: {audio_duration:.2f}s]", flush=True)

            # Prepare context from previous transcriptions
            context_text = " ".join(self.context[-self.max_context_words:]) if self.context else None

            if context_text:
                print(f"[Context: ...{context_text[-100:]}]")  # Show last 100 chars of context

            # Transcribe directly from buffer with context
            segments, _ = self.model.transcribe(
                audio_buffer,
                beam_size=5,
                language='en',
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

            if segments_list:
                first_segment = segments_list[0]
                last_segment = segments_list[-1]

                print(f"[VAD: Speech detected from {first_segment.start:.2f}s to {last_segment.end:.2f}s]", flush=True)
                print(f"[VAD: Trimmed {first_segment.start:.2f}s from start, {audio_duration - last_segment.end:.2f}s from end]", flush=True)

                for i, segment in enumerate(segments_list):
                    print(f"[Segment {i+1}: {segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}", flush=True)
                    words = segment.text.strip().split()
                    new_text_words.extend(words)
            else:
                print(f"[VAD: No speech detected in {audio_duration:.2f}s buffer]", flush=True)

            # Update context with new words
            self.context.extend(new_text_words)

            transcription = " ".join(new_text_words)
            return transcription.strip()
        except Exception as e:
            print(f"GPU transcription failed: {e}")
            return ""

    def reset_context(self):
        """Clear context without restarting model"""
        self.context = []


# ============================================================================
# MAIN DAEMON
# ============================================================================

class PaDCDaemon:
    def __init__(self):
        self.state = State.IDLE
        self.recorder = AudioRecorder()
        self.whisper = None
        self.running = True
        self.recording_mode = RecordingMode.NORMAL
        self.is_processing = False
        self.context_reset_timer = None

        # Initialize status file
        self._update_status_file()

        # Initialize GPU Whisper model (heavy operation, done once at startup)
        self._init_whisper()

        # Signal handlers
        signal.signal(signal.SIGTERM, lambda s, f: self.shutdown())
        signal.signal(signal.SIGINT, lambda s, f: self.shutdown())

        self.start_recording()

    def _init_whisper(self):
        """Initialize GPU Whisper model - runs once at startup"""
        model_size = os.environ.get("PADC_MODEL", os.environ.get("WHISPER_MODEL", "base"))
        print(f"[{time.strftime('%H:%M:%S')}] Loading GPU Whisper model ({model_size})...")

        self.whisper = GPUWhisperModel(model_size=model_size)
        print(f"[{time.strftime('%H:%M:%S')}] GPU Whisper model loaded successfully")

    def _auto_reset_context(self):
        """Auto-reset context after inactivity"""
        if self.whisper:
            self.whisper.reset_context()
            print(f"[{time.strftime('%H:%M:%S')}] Context auto-reset (1min inactivity)", flush=True)

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

        # Reset context immediately
        if self.whisper:
            self.whisper.reset_context()

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

            # Transcribe directly from buffer (no temp file!)
            text = self.whisper.transcribe_buffer(audio_buffer)

            # Process text to fix common Whisper mistakes
            text = self.process_text(text)

            total_time = time.time() - processing_start_time

            if text:
                # Log transcription to file
                self._log_transcription(text)

                clipcontent = pyperclip.paste()
                pyperclip.copy(text)


                # Handle insert modes
                if recording_mode == RecordingMode.INSERT:
                    self._insert_with_xdotool(text)
                    time.sleep(0.5)  # Wait before restoring clipboard to prevent race conditions
                    pyperclip.copy(clipcontent)
                    print(f"... transcribed ({total_time:.2f}s)\n ✓ Pasted: {text}", flush=True)
                elif recording_mode == RecordingMode.INSERT_CONTINUE:
                    self._insert_with_xdotool(text)
                    time.sleep(0.5)  # Wait before restoring clipboard to prevent race conditions
                    pyperclip.copy(clipcontent)
                    print(f"... transcribed ({total_time:.2f}s)\n ✓ Pasted: {text}", flush=True)
                else:
                    print(f"... transcribed ({total_time:.2f}s):\n ✓ Copied: {text}", flush=True)
            else:
                print("... no speech detected", flush=True)

        except Exception as e:
            print(f"... transcription error: {e}", flush=True)
        finally:
            self.is_processing = False
            self._update_status_file()
            # Start context reset timer after every transcription
            # Will be cancelled if user starts recording again within 1 minute
            self._start_context_reset_timer()

    def _insert_with_xdotool(self, text):
        """Paste using Shift+Insert"""
        try:
            subprocess.run(["xdotool", "key", "shift+Insert"], check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass  # Error will be handled in calling function

    def insert(self):
        """Stop recording and insert (no toggle, no auto-restart)"""
        if self.state != State.RECORDING:
            return "not_recording"

        # Play cancel sound in a separate thread to not block
        threading.Thread(target=self.recorder.play_cancel_sound).start()

        self.recording_mode = RecordingMode.INSERT
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

    def reset_context(self):
        """Reset transcription context"""
        if self.whisper:
            self.whisper.reset_context()
            return "context_reset"
        return "whisper_not_initialized"

    def process_command(self, cmd):
        """Process a single command - cancel, insert, toggle-insert, toggle-insert-continue, reset-context"""
        cmd = cmd.strip().lower()

        if cmd == "insert":
            return self.insert()
        elif cmd == "toggle-insert":
            return self.toggle_insert()
        elif cmd == "toggle-insert-continue":
            return self.toggle_insert_continue()
        elif cmd == "cancel":
            return self.cancel_recording()
        elif cmd == "reset-context":
            return self.reset_context()
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
