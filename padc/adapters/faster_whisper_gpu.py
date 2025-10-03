import os
import sys
from pathlib import Path
from typing import Optional
from .base import STTAdapter


class FasterWhisperGPUAdapter(STTAdapter):
    def __init__(self, model_size: str = "base"):
        self.model_size = model_size
        self.model = None
        self.device = "cpu"
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
        from faster_whisper import WhisperModel
        
        # Check if CUDA is actually available and working
        if self._check_cuda_available():
            try:
                # Try GPU with int8 first (more compatible)
                self.model = WhisperModel(
                    self.model_size,
                    device="cuda",
                    compute_type="int8"
                )
                self.device = "cuda"
                self.compute_type = "int8"
                print(f"Initialized Faster Whisper GPU model: {self.model_size} (int8)")
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
                    self.device = "cuda"
                    self.compute_type = "float16"
                    print(f"Initialized Faster Whisper GPU model: {self.model_size} (float16)")
                    return
                except Exception as e2:
                    print(f"Failed with float16 on GPU: {e2}")
        
        # No fallback - exit on failure
        print("CUDA not available or failed")
        raise ValueError()
        sys.exit(1)

    def transcribe(self, audio_path: Path, language: Optional[str] = None) -> str:
        if not self.model:
            raise RuntimeError("Faster Whisper GPU model not initialized")
        
        try:
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
        except Exception as e:
            print(f"GPU transcription failed: {e}")
            sys.exit(1)

    def is_available(self) -> bool:
        return self.model is not None
