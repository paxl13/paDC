#!/usr/bin/env python3
"""paDC daemon - GPU-only buffer-based speech-to-text"""

import os
import sys
import signal
import time
import threading
import subprocess
from datetime import datetime
from enum import Enum

import numpy as np
import pyperclip

from .config import (
    DaemonConfig,
    FIFO_PATH,
    PID_FILE,
    STATUS_FILE,
    TRANSCRIPTION_LOG,
    DEBUG_AUDIO_DIR,
)
from .audio import (
    AudioRecorder,
    MicGainController,
    normalize_audio_buffer,
    save_audio_buffer_to_wav,
)
from .whisper import GPUWhisperModel, process_text, is_hallucination


class State(Enum):
    IDLE = "idle"
    RECORDING = "recording"


class RecordingMode(Enum):
    NORMAL = "normal"  # Only used for cancel
    INSERT = "insert"  # Paste with Shift+Insert
    INSERT_CONTINUE = "insert_continue"  # Paste with Shift+Insert, then auto-restart
    CLAUDE_SEND = "claude_send"  # Send to Claude via claude-send-active script


class PaDCDaemon:
    def __init__(self, config: DaemonConfig):
        self.config = config
        self.state = State.IDLE
        self.recorder = AudioRecorder(config.audio)
        self.whisper_gpu = None
        self.running = True
        self.recording_mode = RecordingMode.NORMAL
        self.is_processing = False

        # Log configuration
        if config.audio.normalize_target > 0.0:
            print(f"[{time.strftime('%H:%M:%S')}] Audio normalization enabled (target: {config.audio.normalize_target:.1%})")
        else:
            print(f"[{time.strftime('%H:%M:%S')}] Audio normalization disabled (using raw levels)")

        if config.audio.debug_save_audio:
            print(f"[{time.strftime('%H:%M:%S')}] Debug audio saving enabled -> {DEBUG_AUDIO_DIR}")

        # Real-time mode state
        self.realtime_mode = False
        self.realtime_thread = None
        self.realtime_recording_mode = RecordingMode.CLAUDE_SEND
        self.last_speech_time = None
        self.last_transcription_time = None
        self.has_speech_since_last_transcription = False

        # Hardware gain control
        self.mic_gain = None
        if config.hw_gain.enabled:
            self.mic_gain = MicGainController(config.hw_gain)
            if self.mic_gain.audio_system:
                print(f"[{time.strftime('%H:%M:%S')}] HW gain control: {self.mic_gain.audio_system}, current: {self.mic_gain.current_gain:.0f}%")
            else:
                print(f"[{time.strftime('%H:%M:%S')}] HW gain control: disabled (no pactl/wpctl)")

        # Initialize status file
        self._update_status_file()

        # Initialize GPU Whisper model
        self._init_whisper_model()

        # Signal handlers
        signal.signal(signal.SIGTERM, lambda s, f: self.shutdown())
        signal.signal(signal.SIGINT, lambda s, f: self.shutdown())

        self.start_recording()

    def _init_whisper_model(self):
        """Initialize GPU Whisper model - runs once at startup"""
        print(f"[{time.strftime('%H:%M:%S')}] Loading GPU Whisper model ({self.config.whisper.model_size}, language: {self.config.whisper.language})...")
        self.whisper_gpu = GPUWhisperModel(self.config.whisper)
        print(f"[{time.strftime('%H:%M:%S')}] GPU Whisper model loaded successfully")

    def _update_status_file(self):
        """Update status file for tmux status bar"""
        try:
            if self.is_processing:
                status = "#[bg=yellow]process#[default]"
            elif self.realtime_mode:
                if self.realtime_recording_mode == RecordingMode.INSERT:
                    status = "#[bg=green]realtime-insert#[default]"
                else:
                    status = "#[bg=green]realtime#[default]"
            elif self.state == State.RECORDING:
                status = "online"
            else:
                status = "standby"
            STATUS_FILE.write_text(status)
        except Exception:
            pass

    def _log_transcription(self, text: str):
        """Log transcription to file with date and time"""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(TRANSCRIPTION_LOG, "a", encoding="utf-8") as f:
                f.write(f"{timestamp} {text}\n")
        except Exception:
            pass

    def start_recording(self):
        """Start recording audio"""
        if self.state == State.RECORDING:
            return "already_recording"

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
        """Drop audio buffer while continuing to record (keeps context)

        Double-cancel within 1 second resets the transcription context.
        """
        if self.state != State.RECORDING:
            return "not_recording"

        # Auto-disable realtime mode
        if self.realtime_mode:
            self.realtime_mode = False
            self._update_status_file()
            threading.Thread(target=self.recorder.play_realtime_off_chime).start()

        current_time = time.time()

        # Check if this is a double-cancel (within 1 second of last cancel)
        if hasattr(self, '_last_cancel_time') and (current_time - self._last_cancel_time) < 1.0:
            # Double cancel: reset context
            if self.whisper_gpu:
                self.whisper_gpu.reset_context()
            self._last_cancel_time = 0  # Reset to prevent triple-cancel
            print(f"[{time.strftime('%H:%M:%S')}] Context reset (double-cancel)", flush=True)
            return "buffer_cleared"

        # Single cancel: clear buffer and reset detected language
        self.recorder.audio_data.clear()
        self._last_cancel_time = current_time

        # Reset detected language (but keep context tokens)
        if self.whisper_gpu:
            self.whisper_gpu.detected_language = None

        # Reset mode to normal
        self.recording_mode = RecordingMode.NORMAL

        print(f"[{time.strftime('%H:%M:%S')}] Buffer cleared (still recording)", flush=True)
        return "buffer_cleared"

    def _realtime_monitor_thread(self):
        """Monitor audio buffer for silence or max interval trigger"""
        while self.realtime_mode and self.running:
            time.sleep(0.5)

            if not self.recorder.audio_data or self.is_processing:
                continue

            current_time = time.time()

            # Analyser les derniers chunks pour silence
            recent_chunks = list(self.recorder.audio_data)[-10:]  # ~0.6s de donnees
            if recent_chunks:
                combined = np.concatenate(recent_chunks)
                rms = np.sqrt(np.mean(combined ** 2))
                is_silent = rms < self.config.realtime.rms_threshold

                if is_silent:
                    silence_duration = current_time - self.last_speech_time if self.last_speech_time else 0
                    if self.has_speech_since_last_transcription and self.last_speech_time and silence_duration >= self.config.realtime.silence_threshold:
                        self._trigger_realtime_transcription(f"silence ({silence_duration:.1f}s)")
                else:
                    # Speech detected - reset silence countdown and mark that we have speech
                    self.last_speech_time = current_time
                    self.has_speech_since_last_transcription = True

            # Check max interval
            if self.last_transcription_time:
                interval = current_time - self.last_transcription_time
                if interval >= self.config.realtime.max_interval:
                    self._trigger_realtime_transcription(f"max interval ({interval:.1f}s)")

    def _trigger_realtime_transcription(self, reason: str = "manual"):
        """Trigger transcription in real-time mode"""
        if self.is_processing:
            return

        print(f"[{time.strftime('%H:%M:%S')}] Realtime trigger: {reason}", flush=True)
        self.last_transcription_time = time.time()
        self.last_speech_time = time.time()
        self.has_speech_since_last_transcription = False
        self.recording_mode = self.realtime_recording_mode
        self.stop_recording()

    def _transcribe(self, audio_buffer, processing_start_time, recording_mode):
        """Transcribe audio in background thread - direct buffer transcription"""
        try:
            if audio_buffer is None or audio_buffer.size == 0:
                print("... no audio captured", flush=True)
                return

            # Apply audio normalization if enabled
            norm_info = {}
            if self.config.audio.normalize_target > 0.0:
                audio_buffer, norm_info = normalize_audio_buffer(audio_buffer, self.config.audio.normalize_target)

            # Debug: Save audio buffer to WAV file if enabled (after normalization)
            if self.config.audio.debug_save_audio:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                debug_wav_path = DEBUG_AUDIO_DIR / f"buffer_{timestamp}.wav"
                save_audio_buffer_to_wav(audio_buffer, debug_wav_path, sample_rate=16000)

            # Transcribe with GPU
            text, info = self.whisper_gpu.transcribe_buffer(audio_buffer)

            # Process text to fix common Whisper mistakes
            text, marked_text = process_text(text)

            # Check for hallucinations and filter them out
            if text and is_hallucination(text):
                timestamp = time.strftime('%H:%M:%S')
                total_time = time.time() - processing_start_time
                print(f"\n[{timestamp}]\n┌─ Hallucination filtered ({total_time:.2f}s)\n│ Buffer: {info.get('buffer_duration', 0):.2f}s\n└─ Ignored: \"{text}\"", flush=True)
                return

            total_time = time.time() - processing_start_time

            # Build comprehensive log output
            if text:
                # Log transcription to file
                self._log_transcription(text)

                # Adjust hardware gain only when speech was detected
                hw_gain_info = {}
                if self.mic_gain and norm_info.get('peak_before'):
                    hw_gain_info = self.mic_gain.adjust_for_peak(norm_info['peak_before'])

                # Display consolidated transcription info
                timestamp = time.strftime('%H:%M:%S')
                log_lines = []
                log_lines.append(f"\n[{timestamp}]")

                log_lines.append(f"┌─ Transcription ({total_time:.2f}s)")
                log_lines.append(f"│ Buffer: {info.get('buffer_duration', 0):.2f}s")

                # Show normalization info if applied
                if norm_info.get('normalized'):
                    gain_db = norm_info.get('gain_db', 0)
                    peak_before = norm_info.get('peak_before', 0)
                    peak_after = norm_info.get('peak_after', 0)
                    clipping = " [CLIPPED]" if norm_info.get('clipping_occurred') else ""
                    log_lines.append(f"│ Gain: {gain_db:+.1f}dB (peak: {peak_before:.1%} → {peak_after:.1%}){clipping}")

                # Show hardware gain adjustment if it occurred
                if hw_gain_info.get('adjusted'):
                    old_hw = hw_gain_info['old_gain']
                    new_hw = hw_gain_info['new_gain']
                    reason = hw_gain_info['reason']
                    log_lines.append(f"│ HW Gain: {old_hw:.0f}% → {new_hw:.0f}% ({reason})")

                # Show speech info
                if info.get('has_speech'):
                    log_lines.append(f"│ Speech: {info.get('speech_start', 0):.2f}s → {info.get('speech_end', 0):.2f}s")
                    log_lines.append(f"│ VAD trimmed: {info.get('trimmed_start', 0):.2f}s (start), {info.get('trimmed_end', 0):.2f}s (end)")
                    lang_hint = info.get('language_hint', '?')
                    lang_detected = info.get('language', '?')
                    lang_prob = info.get('language_probability', 0) * 100
                    log_lines.append(f"│ Language: {lang_hint} → {lang_detected} ({lang_prob:.0f}%)")
                log_lines.append(f"│ Context: {info.get('context_tokens_before', 0)} → {info.get('context_tokens_after', 0)} tokens")

                log_lines.append(f"│")

                # Show segments if available
                if info.get('segments'):
                    for seg in info['segments']:
                        log_lines.append(f"│ [{seg['start']:.2f}s-{seg['end']:.2f}s]{seg['text']}")

                log_lines.append(f"│")

                # Show result - use marked_text if fillers were removed to show [] inline
                if marked_text:
                    log_lines.append(f"│ Result: {marked_text}")
                else:
                    log_lines.append(f"│ Result: {text}")

                # Handle clipboard and insert modes
                if recording_mode == RecordingMode.CLAUDE_SEND:
                    # CLAUDE_SEND mode: don't touch clipboard, just send to script
                    insert_method = self._insert_with_claude_send(text)
                    log_lines.append(f"└─ ✓ Sent to Claude ({insert_method})")
                else:
                    # INSERT modes: copy to clipboard and paste
                    clipcontent = pyperclip.paste()
                    pyperclip.copy(text + " ")

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
            subprocess.run(
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

        # Auto-disable realtime mode
        if self.realtime_mode:
            self.realtime_mode = False
            self._update_status_file()
            threading.Thread(target=self.recorder.play_realtime_off_chime).start()

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

        # Auto-disable realtime mode
        if self.realtime_mode:
            self.realtime_mode = False
            self._update_status_file()
            threading.Thread(target=self.recorder.play_realtime_off_chime).start()

        return self.toggle()

    def toggle_insert_continue(self):
        """Toggle recording with insert-continue mode"""
        self.recording_mode = RecordingMode.INSERT_CONTINUE

        # Auto-disable realtime mode
        if self.realtime_mode:
            self.realtime_mode = False
            self._update_status_file()
            threading.Thread(target=self.recorder.play_realtime_off_chime).start()

        return self.toggle()

    def toggle(self):
        """Toggle recording state"""
        if self.state == State.IDLE:
            return self.start_recording()
        else:
            return self.stop_recording()

    def _toggle_realtime_with_mode(self, mode: RecordingMode, mode_name: str):
        """Toggle real-time transcription mode with specified recording mode"""
        if self.realtime_mode:
            # Turn off
            self.realtime_mode = False
            if self.realtime_thread:
                self.realtime_thread.join(timeout=1.0)
                self.realtime_thread = None

            # Trigger final transcription if there's speech in buffer
            if self.has_speech_since_last_transcription and self.recorder.audio_data:
                print(f"[{time.strftime('%H:%M:%S')}] Realtime trigger: toggle-off", flush=True)
                self.recording_mode = self.realtime_recording_mode
                self.stop_recording()

            threading.Thread(target=self.recorder.play_realtime_off_chime).start()

            print(f"[{time.strftime('%H:%M:%S')}] Real-time mode: OFF")
        else:
            # Turn on
            self.realtime_mode = True
            self.realtime_recording_mode = mode

            # Check existing buffer for speech before flushing
            if self.recorder.audio_data:
                buffer_chunks = list(self.recorder.audio_data)
                combined = np.concatenate(buffer_chunks)
                rms = np.sqrt(np.mean(combined ** 2))
                duration = len(combined) / 16000  # 16kHz sample rate

                # Only transcribe if: speech detected (RMS above threshold) AND duration >= 1s
                if rms >= self.config.realtime.rms_threshold and duration >= 1.0:
                    print(f"[{time.strftime('%H:%M:%S')}] Realtime trigger: startup (buffer: {duration:.1f}s, RMS: {rms:.4f})", flush=True)
                    self.recording_mode = mode
                    self.stop_recording()
                else:
                    # Buffer is silence or too short - discard it
                    print(f"[{time.strftime('%H:%M:%S')}] Realtime startup: buffer ignored (duration: {duration:.1f}s, RMS: {rms:.4f})", flush=True)
                    self.recorder.audio_data.clear()

            self.last_speech_time = time.time()
            self.last_transcription_time = time.time()
            self.has_speech_since_last_transcription = False
            self.realtime_thread = threading.Thread(target=self._realtime_monitor_thread, daemon=True)
            self.realtime_thread.start()
            # Chime distinct pour INSERT vs CLAUDE_SEND
            if mode == RecordingMode.INSERT:
                threading.Thread(target=self.recorder.play_realtime_insert_on_chime).start()
            else:
                threading.Thread(target=self.recorder.play_realtime_on_chime).start()
            print(f"[{time.strftime('%H:%M:%S')}] Real-time mode: ON ({mode_name})")

        self._update_status_file()
        return "realtime_on" if self.realtime_mode else "realtime_off"

    def toggle_realtime(self):
        """Toggle real-time transcription mode (CLAUDE_SEND)"""
        return self._toggle_realtime_with_mode(RecordingMode.CLAUDE_SEND, "claude")

    def toggle_realtime_insert(self):
        """Toggle real-time transcription mode (INSERT - Shift+Insert)"""
        return self._toggle_realtime_with_mode(RecordingMode.INSERT, "insert")

    def toggle_realtime_clear(self):
        """Clear buffer when activating real-time mode (CLAUDE_SEND)"""
        if not self.realtime_mode:
            self.recorder.audio_data.clear()
            print(f"[{time.strftime('%H:%M:%S')}] Buffer cleared before realtime start", flush=True)
        return self._toggle_realtime_with_mode(RecordingMode.CLAUDE_SEND, "claude")

    def toggle_realtime_insert_clear(self):
        """Clear buffer when activating real-time mode (INSERT)"""
        if not self.realtime_mode:
            self.recorder.audio_data.clear()
            print(f"[{time.strftime('%H:%M:%S')}] Buffer cleared before realtime start", flush=True)
        return self._toggle_realtime_with_mode(RecordingMode.INSERT, "insert")

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

    def reset_context(self):
        """Reset transcription context"""
        if self.whisper_gpu:
            self.whisper_gpu.reset_context()
        return "context_reset"

    def process_command(self, cmd):
        """Process commands"""
        cmd = cmd.strip().lower()

        if cmd == "insert":
            return self.insert()
        elif cmd == "toggle-insert":
            return self.toggle_insert()
        elif cmd == "toggle-insert-continue":
            return self.toggle_insert_continue()
        elif cmd == "claude-send":
            return self.claude_send()
        elif cmd == "toggle-realtime":
            return self.toggle_realtime()
        elif cmd == "toggle-realtime-insert":
            return self.toggle_realtime_insert()
        elif cmd == "toggle-realtime-clear":
            return self.toggle_realtime_clear()
        elif cmd == "toggle-realtime-insert-clear":
            return self.toggle_realtime_insert_clear()
        elif cmd == "reset-context":
            return self.reset_context()
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
                        if response and response not in ["recording_started", "processing", "recording_cancelled", "buffer_cleared"]:
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

    config = DaemonConfig.from_env()
    daemon = PaDCDaemon(config)
    daemon.run()


if __name__ == "__main__":
    main()
