import os
from pathlib import Path
from typing import Optional
from openai import OpenAI
from .base import STTAdapter


class OpenAIAdapter(STTAdapter):
    def __init__(self, api_key: Optional[str] = None, model: str = "whisper-1"):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.client = None
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key)

    def transcribe(self, audio_path: Path, language: Optional[str] = None) -> str:
        if not self.client:
            raise RuntimeError("OpenAI client not initialized. Please provide API key.")
        
        with open(audio_path, "rb") as audio_file:
            params = {
                "model": self.model,
                "file": audio_file,
                "response_format": "text"
            }
            if language:
                params["language"] = language
            
            transcription = self.client.audio.transcriptions.create(**params)
        
        return transcription.strip()

    def is_available(self) -> bool:
        return self.client is not None