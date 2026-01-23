"""Module audio: enregistrement, normalisation, chimes, controle gain"""

import queue
import re
import subprocess
import threading
import time
import wave
from collections import deque
from pathlib import Path

import numpy as np
import sounddevice as sd

from .config import AudioConfig, HardwareGainConfig


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

    # Final safety check: hard clip at +/-1.0 (should rarely trigger)
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


class MicGainController:
    """Adaptive hardware microphone gain control via PulseAudio/PipeWire"""

    DEFAULT_INITIAL_GAIN = 50.0  # Start at 50% if no saved gain

    def __init__(self, config: HardwareGainConfig):
        self.min_gain = config.min_gain
        self.max_gain = config.max_gain
        self.target_peak = config.target_peak
        self.max_adjustment = 0.1  # 10% max adjustment per iteration
        # Target window: 80% to 120% of target
        self.low_ratio = 0.8
        self.high_ratio = 1.2
        self.audio_system = self._detect_system()
        self._init_gain()

    def _detect_system(self) -> str | None:
        """Detect available audio system (pactl or wpctl)"""
        for cmd in ['pactl', 'wpctl']:
            try:
                subprocess.run([cmd, '--version'], capture_output=True, check=True)
                return cmd
            except (FileNotFoundError, subprocess.CalledProcessError):
                continue
        return None

    def _init_gain(self):
        """Initialize gain: set to default 50% at startup"""
        if not self.audio_system:
            self.current_gain = None
            return

        # Always start at default gain (50%)
        self.set_gain(self.DEFAULT_INITIAL_GAIN)
        self.current_gain = self.DEFAULT_INITIAL_GAIN

    def get_gain(self) -> float | None:
        """Read current microphone gain from system"""
        if not self.audio_system:
            return None

        try:
            if self.audio_system == 'pactl':
                result = subprocess.run(
                    ['pactl', 'get-source-volume', '@DEFAULT_SOURCE@'],
                    capture_output=True, text=True, check=True
                )
                match = re.search(r'(\d+)%', result.stdout)
                if match:
                    return float(match.group(1))
            elif self.audio_system == 'wpctl':
                result = subprocess.run(
                    ['wpctl', 'get-volume', '@DEFAULT_AUDIO_SOURCE@'],
                    capture_output=True, text=True, check=True
                )
                match = re.search(r'[\d.]+', result.stdout)
                if match:
                    return float(match.group()) * 100.0
        except Exception:
            pass
        return None

    def set_gain(self, percent: float) -> bool:
        """Set microphone hardware gain"""
        if not self.audio_system:
            return False

        try:
            if self.audio_system == 'pactl':
                subprocess.run(
                    ['pactl', 'set-source-volume', '@DEFAULT_SOURCE@', f'{percent:.0f}%'],
                    check=True, capture_output=True
                )
            elif self.audio_system == 'wpctl':
                linear = percent / 100.0
                subprocess.run(
                    ['wpctl', 'set-volume', '@DEFAULT_AUDIO_SOURCE@', f'{linear:.2f}'],
                    check=True, capture_output=True
                )
            return True
        except subprocess.CalledProcessError:
            return False

    def adjust_for_peak(self, measured_peak: float) -> dict:
        """Adjust hardware gain based on measured peak level (before normalization)

        Args:
            measured_peak: Peak audio level before normalization (0.0 to 1.0)

        Returns:
            dict with keys: adjusted, old_gain, new_gain, reason
        """
        if not self.audio_system or measured_peak < 1e-6:
            return {'adjusted': False}

        old_gain = self.current_gain if self.current_gain else self.get_gain()
        if old_gain is None:
            return {'adjusted': False}

        new_gain = old_gain
        reason = None

        # Target window boundaries
        low_threshold = self.target_peak * self.low_ratio
        high_threshold = self.target_peak * self.high_ratio

        if measured_peak < low_threshold:
            # Peak too low -> increase gain
            distance_ratio = low_threshold / measured_peak
            adjustment_pct = min((distance_ratio - 1) * 0.1, self.max_adjustment)
            adjustment = adjustment_pct * old_gain
            new_gain = min(old_gain + adjustment, self.max_gain)
            reason = f"peak bas ({measured_peak:.0%})"

        elif measured_peak > high_threshold:
            # Peak too high -> decrease gain
            distance_ratio = measured_peak / high_threshold
            adjustment_pct = min((distance_ratio - 1) * 0.1, self.max_adjustment)
            adjustment = adjustment_pct * old_gain
            new_gain = max(old_gain - adjustment, self.min_gain)
            reason = f"peak haut ({measured_peak:.0%})"

        if new_gain != old_gain:
            if self.set_gain(new_gain):
                self.current_gain = new_gain
                return {'adjusted': True, 'old_gain': old_gain, 'new_gain': new_gain, 'reason': reason}

        return {'adjusted': False}


class AudioRecorder:
    """Audio recorder - records audio to numpy buffer"""

    def __init__(self, config: AudioConfig):
        self.sample_rate = config.sample_rate
        self.channels = config.channels
        self.buffer_seconds = config.buffer_seconds
        self.recording = False
        self.audio_queue = queue.Queue()

        # We'll calculate maxlen dynamically after seeing first chunk
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
        sd.wait()

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
        sd.wait()

    def play_realtime_on_chime(self):
        """Double beep ascendant pour indiquer activation real-time"""
        duration = 0.1
        for freq in [440, 660]:  # A4 -> E5 (quinte ascendante)
            t = np.linspace(0, duration, int(self.sample_rate * duration))
            envelope = np.exp(-5 * t)
            tone = envelope * np.sin(2 * np.pi * freq * t) * 0.2
            sd.play(tone, self.sample_rate)
            sd.wait()

    def play_realtime_insert_on_chime(self):
        """Triple beep rapide ascendant pour real-time INSERT mode"""
        duration = 0.08
        for freq in [523, 659, 784]:  # C5 -> E5 -> G5 (accord majeur ascendant)
            t = np.linspace(0, duration, int(self.sample_rate * duration))
            envelope = np.exp(-6 * t)
            tone = envelope * np.sin(2 * np.pi * freq * t) * 0.2
            sd.play(tone, self.sample_rate)
            sd.wait()

    def play_realtime_off_chime(self):
        """Double beep descendant pour indiquer desactivation"""
        duration = 0.1
        for freq in [660, 440]:  # E5 -> A4 (quinte descendante)
            t = np.linspace(0, duration, int(self.sample_rate * duration))
            envelope = np.exp(-5 * t)
            tone = envelope * np.sin(2 * np.pi * freq * t) * 0.2
            sd.play(tone, self.sample_rate)
            sd.wait()

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
