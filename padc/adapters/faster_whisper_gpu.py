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
        
        # Fallback to CPU
        print("CUDA not available or failed, using CPU")
        try:
            self.model = WhisperModel(
                self.model_size,
                device="cpu",
                compute_type="int8"
            )
            self.device = "cpu"
            self.compute_type = "int8"
            print(f"Using CPU for Faster Whisper: {self.model_size}")
        except Exception as cpu_error:
            print(f"Failed to initialize Faster Whisper model: {cpu_error}")

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
            # If GPU fails during transcription, try to reinitialize with CPU
            if self.device == "cuda":
                print(f"GPU transcription failed: {e}")
                print("Attempting CPU fallback...")
                from faster_whisper import WhisperModel
                try:
                    self.model = WhisperModel(
                        self.model_size,
                        device="cpu",
                        compute_type="int8"
                    )
                    self.device = "cpu"
                    self.compute_type = "int8"
                    
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
                except Exception as cpu_error:
                    raise RuntimeError(f"Both GPU and CPU transcription failed: {cpu_error}")
            else:
                raise

    def is_available(self) -> bool:
        return self.model is not None