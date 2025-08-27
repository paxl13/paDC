#!/usr/bin/env python3
"""Ultra-minimal paDC daemon - keeps model loaded, responds to FIFO commands"""

import os
import sys
import signal
import time
import threading
from pathlib import Path
from enum import Enum
import numpy as np
import pyperclip
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from padc.audio import AudioRecorder
from padc.adapters import FasterWhisperAdapter, FasterWhisperGPUAdapter, OpenAIAdapter

load_dotenv()

# Paths
FIFO_PATH = Path("/tmp/padc.fifo")
PID_FILE = Path("/tmp/padc_daemon.pid")
LOG_FILE = Path("/tmp/padc_daemon.log")


class State(Enum):
    IDLE = "idle"
    RECORDING = "recording"


class RecordingMode(Enum):
    NORMAL = "normal"
    PASTE = "paste"  # Will type text with xdotool
    INSERT = "insert"  # Will paste with Shift+Insert
    INSERT_ENTER = "insert_enter"  # Will paste with Shift+Insert, then press Enter


class PaDCDaemon:
    def __init__(self):
        self.state = State.IDLE
        self.recorder = AudioRecorder()
        self.audio_buffer = None
        self.adapter = None
        self.running = True
        self.recording_mode = RecordingMode.NORMAL

        # Initialize adapter once (heavy operation)
        self._init_adapter()

        # Signal handlers
        signal.signal(signal.SIGTERM, lambda s, f: self.shutdown())
        signal.signal(signal.SIGINT, lambda s, f: self.shutdown())

    def _init_adapter(self):
        """Initialize STT adapter - runs once at startup"""
        adapter_type = os.environ.get("PADC_ADAPTER", "local").lower()
        model_size = os.environ.get(
            "PADC_MODEL", os.environ.get("WHISPER_MODEL", "base")
        )

        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loading {adapter_type} adapter with {model_size} model..."
        )

        try:
            if adapter_type == "local_gpu":
                # Set CUDA library path if needed
                venv_path = Path(__file__).parent / ".venv"
                if venv_path.exists():
                    cuda_lib = (
                        venv_path / "lib/python3.12/site-packages/nvidia/cudnn/lib"
                    )
                    if cuda_lib.exists():
                        os.environ["LD_LIBRARY_PATH"] = (
                            f"{cuda_lib}:{os.environ.get('LD_LIBRARY_PATH', '')}"
                        )
                self.adapter = FasterWhisperGPUAdapter(model_size=model_size)
            elif adapter_type == "openai":
                api_key = os.environ.get("OPENAI_API_KEY")
                if api_key:
                    self.adapter = OpenAIAdapter(api_key=api_key)
                else:
                    print(
                        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] No OpenAI key, falling back to local"
                    )
                    self.adapter = FasterWhisperAdapter(model_size=model_size)
            else:  # local
                self.adapter = FasterWhisperAdapter(model_size=model_size)

            print(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Adapter loaded: {self.adapter.__class__.__name__}"
            )
        except Exception as e:
            print(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Failed to load adapter: {e}, using fallback"
            )
            self.adapter = FasterWhisperAdapter(model_size="base")

    def start_recording(self):
        """Start recording audio"""
        if self.state == State.RECORDING:
            return "already_recording"

        self.state = State.RECORDING
        self.recorder.start(play_chime=True)
        return "recording_started"

    def stop_recording(self):
        """Stop recording and transcribe"""
        if self.state != State.RECORDING:
            return "not_recording"

        # Get audio from memory
        self.audio_buffer = self.recorder.stop()
        self.state = State.IDLE
        self.processing_start_time = time.time()  # Track when processing started

        # Process in background
        threading.Thread(target=self._transcribe, daemon=True).start()
        return "processing"

    def _transcribe(self):
        """Transcribe audio in background thread"""
        try:
            if self.audio_buffer is None or self.audio_buffer.size == 0:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] No audio captured")
                return

            # Save to temporary file (adapters need file path for now)
            import tempfile
            import scipy.io.wavfile as wavfile

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                audio_int16 = (self.audio_buffer * 32767).astype(np.int16)
                wavfile.write(tmp.name, self.recorder.sample_rate, audio_int16)
                tmp_path = Path(tmp.name)

            # Transcribe
            transcribe_start = time.time()
            text = self.adapter.transcribe(tmp_path)
            transcribe_time = time.time() - transcribe_start
            tmp_path.unlink()  # Clean up

            if text:
                if self.recording_mode != RecordingMode.PASTE:
                    pyperclip.copy(text)

                total_time = time.time() - self.processing_start_time
                print(
                    f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Transcribed ({total_time:.2f}s): {text}",
                    flush=True,
                )

                # Handle paste/insert modes
                if self.recording_mode == RecordingMode.PASTE:
                    self._paste_with_xdotool(text)
                elif self.recording_mode == RecordingMode.INSERT:
                    self._insert_with_xdotool()
                elif self.recording_mode == RecordingMode.INSERT_ENTER:
                    self._insert_enter_with_xdotool()
            else:
                print(
                    f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] No speech detected",
                    flush=True,
                )

        except Exception as e:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Transcription error: {e}")
        finally:
            self.audio_buffer = None
            self.recording_mode = RecordingMode.NORMAL  # Reset to normal mode

    def _paste_with_xdotool(self, text):
        """Type text using xdotool"""
        try:
            import subprocess

            subprocess.run(["xdotool", "type", text], check=True)
            print(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ✓ Typed with xdotool",
                flush=True,
            )
        except subprocess.CalledProcessError:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Failed to type with xdotool")
        except FileNotFoundError:
            print(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] xdotool not found - install it for paste functionality"
            )

    def _insert_with_xdotool(self):
        """Paste using Shift+Insert"""
        try:
            import subprocess

            subprocess.run(["xdotool", "key", "shift+Insert"], check=True)
            print(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ✓ Pasted with Shift+Insert",
                flush=True,
            )
        except subprocess.CalledProcessError:
            print(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Failed to paste with xdotool"
            )
        except FileNotFoundError:
            print(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] xdotool not found - install it for insert functionality"
            )

    def _insert_enter_with_xdotool(self):
        """Paste using Shift+Insert, then press Enter"""
        try:
            import subprocess

            # First paste with Shift+Insert
            subprocess.run(["xdotool", "key", "shift+Insert"], check=True)
            # Small delay to ensure text is pasted
            time.sleep(0.1)
            # Then press Enter
            subprocess.run(["xdotool", "key", "Return"], check=True)
            print(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ✓ Pasted with Shift+Insert and pressed Enter",
                flush=True,
            )
        except subprocess.CalledProcessError:
            print(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Failed to paste and enter with xdotool"
            )
        except FileNotFoundError:
            print(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] xdotool not found - install it for insert-enter functionality"
            )

    def toggle(self):
        """Toggle recording state"""
        if self.state == State.IDLE:
            return self.start_recording()
        else:
            return self.stop_recording()

    def toggle_paste(self):
        """Toggle recording with paste mode"""
        self.recording_mode = RecordingMode.PASTE
        return self.toggle()

    def toggle_insert(self):
        """Toggle recording with insert mode"""
        self.recording_mode = RecordingMode.INSERT
        return self.toggle()

    def toggle_insert_enter(self):
        """Toggle recording with insert-enter mode"""
        self.recording_mode = RecordingMode.INSERT_ENTER
        return self.toggle()

    def status(self):
        """Get daemon status"""
        adapter_name = self.adapter.__class__.__name__ if self.adapter else "None"
        return f"status|{self.state.value}|{adapter_name}"

    def shutdown(self):
        """Shutdown daemon"""
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Shutting down daemon...")
        self.running = False
        if self.state == State.RECORDING:
            self.recorder.stop()

        # Cleanup
        if FIFO_PATH.exists():
            os.unlink(FIFO_PATH)
        if PID_FILE.exists():
            os.unlink(PID_FILE)

        sys.exit(0)

    def process_command(self, cmd):
        """Process a single command"""
        cmd = cmd.strip().lower()

        if cmd == "toggle":
            return self.toggle()
        elif cmd == "toggle-paste":
            return self.toggle_paste()
        elif cmd == "toggle-insert":
            return self.toggle_insert()
        elif cmd == "toggle-insert-enter":
            return self.toggle_insert_enter()
        elif cmd == "start":
            return self.start_recording()
        elif cmd == "stop":
            return self.stop_recording()
        elif cmd == "status":
            return self.status()
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
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] paDC daemon started (PID: {os.getpid()})",
            flush=True,
        )
        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Listening on {FIFO_PATH}",
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
                        if response:
                            print(
                                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Response: {response}",
                                flush=True,
                            )

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Error: {e}")
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

    # Run daemon
    daemon = PaDCDaemon()
    daemon.run()


if __name__ == "__main__":
    main()
