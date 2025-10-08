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

    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.recording = False
        self.audio_queue = queue.Queue()
        self.audio_data = []
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
        self.audio_data = []
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
                    self.audio_data.append(data)
                except queue.Empty:
                    continue

    def stop(self) -> np.ndarray:
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
    """Inline GPU-only Whisper model wrapper"""

    def __init__(self, model_size: str = "base"):
        self.model_size = model_size
        self.model = None
        self.device = "cuda"
        self.compute_type = "int8"
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
        """Transcribe audio buffer directly (no file I/O)"""
        if not self.model:
            raise RuntimeError("GPU Whisper model not initialized")

        if audio_buffer.size == 0:
            return ""

        try:
            # Flatten to 1D if needed (faster-whisper expects 1D float32 array at 16kHz)
            if audio_buffer.ndim > 1:
                audio_buffer = audio_buffer.flatten()

            # Transcribe directly from buffer
            segments, _ = self.model.transcribe(
                audio_buffer,
                language='en',  # Auto-detect
                beam_size=5,
                condition_on_previous_text=True,
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=500
                )
            )

            transcription = " ".join([segment.text.strip() for segment in segments])
            return transcription.strip()
        except Exception as e:
            print(f"GPU transcription failed: {e}")
            return ""


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

        # Initialize status file
        self._update_status_file()

        # Initialize GPU Whisper model (heavy operation, done once at startup)
        self._init_whisper()

        # Signal handlers
        signal.signal(signal.SIGTERM, lambda s, f: self.shutdown())
        signal.signal(signal.SIGINT, lambda s, f: self.shutdown())

    def _init_whisper(self):
        """Initialize GPU Whisper model - runs once at startup"""
        model_size = os.environ.get("PADC_MODEL", os.environ.get("WHISPER_MODEL", "base"))
        print(f"[{time.strftime('%H:%M:%S')}] Loading GPU Whisper model ({model_size})...")

        self.whisper = GPUWhisperModel(model_size=model_size)
        print(f"[{time.strftime('%H:%M:%S')}] GPU Whisper model loaded successfully")

    def _update_status_file(self):
        """Update status file for tmux status bar"""
        try:
            if self.is_processing:
                status = "#[bg=yellow]process#[default]"
            elif self.state == State.RECORDING:
                status = "#[bg=red] record#[default]"
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

    def start_recording(self):
        """Start recording audio"""
        if self.state == State.RECORDING:
            return "already_recording"

        self.state = State.RECORDING
        self._update_status_file()
        self.recorder.start(play_chime=True)
        print(f"[{time.strftime('%H:%M:%S')}] Recording started... ", end="", flush=True)
        return "recording_started"

    def stop_recording(self):
        """Stop recording and transcribe"""
        if self.state != State.RECORDING:
            return "not_recording"

        # Change state immediately to prevent re-entry
        self.state = State.IDLE

        # Get audio from memory
        audio_buffer = self.recorder.stop()

        # Check if we should auto-restart
        should_auto_restart = self.recording_mode == RecordingMode.INSERT_CONTINUE

        # If auto-restart, start recording immediately (no gap)
        if should_auto_restart:
            self.start_recording()  # Start new recording
            # Keep the mode as INSERT_CONTINUE for next iteration
            self.recording_mode = RecordingMode.INSERT_CONTINUE
        else:
            self._update_status_file()

        self.is_processing = True
        self._update_status_file()
        processing_start_time = time.time()

        # Update the same line with processing message
        print("processing", end="", flush=True)

        # Process in background with a copy of the buffer
        threading.Thread(
            target=self._transcribe,
            args=(audio_buffer, processing_start_time, self.recording_mode),
            daemon=True
        ).start()

        # Reset mode to normal if not auto-restarting
        if not should_auto_restart:
            self.recording_mode = RecordingMode.NORMAL

        return "processing"

    def cancel_recording(self):
        """Cancel recording without transcribing"""
        if self.state != State.RECORDING:
            return "not_recording"

        # Stop recording but don't save the audio
        self.recorder.stop()
        self.state = State.IDLE
        self._update_status_file()
        self.recording_mode = RecordingMode.NORMAL  # Reset to normal mode

        # Play cancel sound in a separate thread to not block
        threading.Thread(target=self.recorder.play_cancel_sound).start()

        # Complete the same line that started with "Recording started..."
        print("cancelled", flush=True)
        return "recording_cancelled"

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

    def process_command(self, cmd):
        """Process a single command - cancel, insert, toggle-insert, toggle-insert-continue"""
        cmd = cmd.strip().lower()

        if cmd == "insert":
            return self.insert()
        elif cmd == "toggle-insert":
            return self.toggle_insert()
        elif cmd == "toggle-insert-continue":
            return self.toggle_insert_continue()
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
