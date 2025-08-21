import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wavfile
from pathlib import Path
from typing import Optional
import threading
import queue
import time


class AudioRecorder:
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

    def start(self):
        self.recording = True
        self.audio_data = []
        self.thread = threading.Thread(target=self._record_thread)
        self.thread.start()

    def _record_thread(self):
        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            callback=self._audio_callback,
            dtype=np.float32
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

    def save_to_wav(self, audio_data: np.ndarray, filepath: Path) -> bool:
        if audio_data.size == 0:
            return False
        
        audio_int16 = (audio_data * 32767).astype(np.int16)
        wavfile.write(str(filepath), self.sample_rate, audio_int16)
        return True

    def record_for_duration(self, duration: float) -> np.ndarray:
        self.start()
        time.sleep(duration)
        return self.stop()