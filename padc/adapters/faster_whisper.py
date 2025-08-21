import os
from pathlib import Path
from typing import Optional
from faster_whisper import WhisperModel
from .base import STTAdapter


class FasterWhisperAdapter(STTAdapter):
    def __init__(self, model_size: str = "base"):
        self.model_size = model_size
        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        try:
            self.model = WhisperModel(
                self.model_size,
                device="cpu",
                compute_type="int8"
            )
        except Exception as e:
            print(f"Failed to initialize Faster Whisper model: {e}")

    def transcribe(self, audio_path: Path, language: Optional[str] = None) -> str:
        if not self.model:
            raise RuntimeError("Faster Whisper model not initialized")
        
        segments, _ = self.model.transcribe(
            str(audio_path),
            language=language,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=500
            )
        )
        
        transcription = " ".join([segment.text.strip() for segment in segments])
        return transcription.strip()

    def is_available(self) -> bool:
        return self.model is not None