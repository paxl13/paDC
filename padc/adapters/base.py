from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional


class STTAdapter(ABC):
    @abstractmethod
    def transcribe(self, audio_path: Path, language: Optional[str] = None) -> str:
        pass

    @abstractmethod
    def is_available(self) -> bool:
        pass