from .base import STTAdapter
from .faster_whisper import FasterWhisperAdapter
from .faster_whisper_gpu import FasterWhisperGPUAdapter
from .openai import OpenAIAdapter

__all__ = ["STTAdapter", "FasterWhisperAdapter", "FasterWhisperGPUAdapter", "OpenAIAdapter"]