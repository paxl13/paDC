"""Module Whisper: transcription GPU, contexte, filtrage hallucinations"""

import re
import sys
import time

import numpy as np

from .config import WhisperConfig, STATUS_FILE


# Whisper hallucination patterns - these are generated when there's no real speech
# Patterns are compiled regex that match the ENTIRE transcription (after strip)
HALLUCINATION_PATTERNS = [
    # Sous-titrage patterns (French broadcast artifacts)
    re.compile(r"^Sous-titrag.*", re.IGNORECASE),
    re.compile(r"^Subtitles by.*", re.IGNORECASE),
    re.compile(r"^Sous-titres par.*", re.IGNORECASE),

    # YouTube-style closing phrases
    re.compile(r"^Merci d'avoir regard[ée].*", re.IGNORECASE),
    re.compile(r"^Thank you for watching.*", re.IGNORECASE),
    re.compile(r"^Please subscribe.*", re.IGNORECASE),
    re.compile(r"^N'oublie[sz]? pas de.*abonn.*", re.IGNORECASE),
    re.compile(r"^Au revoir[\s!.]*$", re.IGNORECASE),

    # Single word/short hallucinations (when alone)
    re.compile(r"^you$", re.IGNORECASE),
    re.compile(r"^Thank you\.?$", re.IGNORECASE),
    re.compile(r"^Merci\.?$", re.IGNORECASE),

    # Punctuation only
    re.compile(r"^\.+$"),

    # Onomatopoeias when they are the ENTIRE transcription
    re.compile(r"^[MmHh]+\.?$"),           # Hmm, mmm, Mmmm, etc.
    re.compile(r"^[Ee]+[Uu]?[Hh]+[Mm]*\.?$"),   # euh, ehm, euhm, ehhh
    re.compile(r"^[Oo][Hh]\.?$"),           # Oh
    re.compile(r"^[Aa][Hh]\.?$"),           # Ah
    re.compile(r"^[Uu]+[Hh]+\.?$"),         # Uh, uhh
]

# Filler words/onomatopoeias to remove from within transcriptions
FILLER_WORDS_PATTERN = re.compile(
    r'\b('
    r'[Ee]+[Uu]+[Hh]+[Mm]*|'   # euh, euuh, euhm, euhhm
    r'[Ee]+[Hh]+[Mm]+|'         # ehm, ehmm, ehhmm
    r'[Hh]+[Mm]+|'              # hm, hmm, hmmm, Hmm
    r'[Mm]+[Hh]+[Mm]*|'         # mhm, mmhm, mhmm
    r'[Uu]+[Hh]+|'              # uh, uhh, uhhh
    r'[Aa]+[Hh]+'               # ah, ahh (when used as filler)
    r')\b'
    r'[.,;:!?\s]*',             # Optional trailing punctuation and space
    re.IGNORECASE
)


def is_hallucination(text: str) -> bool:
    """Check if transcription is a known Whisper hallucination pattern

    Args:
        text: Transcription text to check

    Returns:
        True if the text matches a hallucination pattern, False otherwise
    """
    if not text:
        return False

    text_stripped = text.strip()
    if not text_stripped:
        return False

    for pattern in HALLUCINATION_PATTERNS:
        if pattern.match(text_stripped):
            return True

    return False


def remove_filler_words(text: str) -> tuple[str, str | None]:
    """Remove filler words/onomatopoeias from transcription

    Args:
        text: Transcription text to clean

    Returns:
        Tuple of (cleaned_text, marked_text_for_log or None if no fillers removed)
        marked_text_for_log shows [filler] where fillers were removed
    """
    if not text:
        return text, None

    # Check if there are any filler words
    if not FILLER_WORDS_PATTERN.search(text):
        return text, None

    # Create marked version for logging (shows [filler] where fillers were)
    marked = FILLER_WORDS_PATTERN.sub(r'[\1] ', text)
    marked = re.sub(r'\s+', ' ', marked).strip()
    marked = re.sub(r'^[,;:\s]+', '', marked)

    # Create clean version for actual use
    cleaned = FILLER_WORDS_PATTERN.sub(' ', text)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    cleaned = re.sub(r'^[,;:\s]+', '', cleaned)

    return cleaned, marked


def process_text(text: str) -> tuple[str, str | None]:
    """Process and correct commonly misheard words from Whisper

    Returns:
        Tuple of (processed_text, marked_text_for_log or None)
        marked_text_for_log shows [] where fillers were removed
    """
    if not text:
        return text, None

    # First, remove filler words (euh, hmm, etc.)
    processed, marked_text = remove_filler_words(text)

    if not processed:
        return processed, marked_text

    # Word replacement mappings for common Whisper mistakes
    replacements = {
        "cloud": "Claude",
        "Cloud": "Claude",
    }

    # Apply replacements to both processed and marked text
    for wrong, correct in replacements.items():
        processed = processed.replace(wrong, correct)
        if marked_text:
            marked_text = marked_text.replace(wrong, correct)

    return processed, marked_text


class GPUWhisperModel:
    """GPU-only Whisper model wrapper with contextual transcription"""

    # Languages we trust for adaptive detection (others are likely false positives)
    TRUSTED_LANGUAGES = {"en", "fr"}

    def __init__(self, config: WhisperConfig):
        self.model_size = config.model_size
        self.model = None
        self.device = "cuda"
        self.compute_type = "int8"
        self.context_tokens = []  # Store token IDs directly for efficiency
        self.max_context_tokens = config.max_context_tokens
        self.default_language = config.language
        self.detected_language = None  # Last detected trusted language
        self.tokenizer = None
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
        """Initialize GPU Whisper model and tokenizer - exits on failure"""
        from faster_whisper import WhisperModel

        # Check if CUDA is actually available and working
        if not self._check_cuda_available():
            print("ERROR: CUDA not available or failed to initialize")
            STATUS_FILE.write_text("#[bg=red]ERROR#[default]")
            sys.exit(1)

        try:
            # Try GPU with int8 first (more compatible)
            self.model = WhisperModel(
                self.model_size,
                device="cuda",
                compute_type="int8"
            )
            self.compute_type = "int8"
            self.tokenizer = self.model.hf_tokenizer
            print(f"[{time.strftime('%H:%M:%S')}] Initialized GPU Whisper model: {self.model_size} (int8)")
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
                self.compute_type = "float16"
                self.tokenizer = self.model.hf_tokenizer
                print(f"[{time.strftime('%H:%M:%S')}] Initialized GPU Whisper model: {self.model_size} (float16)")
                return
            except Exception as e2:
                print(f"Failed with float16 on GPU: {e2}")

        # No fallback - exit on failure
        print("ERROR: Failed to initialize GPU Whisper model")
        STATUS_FILE.write_text("#[bg=red]ERROR#[default]")
        sys.exit(1)

    def transcribe_buffer(self, audio_buffer: np.ndarray) -> tuple[str, dict]:
        """Transcribe audio buffer directly with contextual awareness

        Returns:
            tuple: (transcription_text, info_dict) where info_dict contains timing and VAD info
        """
        if not self.model:
            raise RuntimeError("GPU Whisper model not initialized")

        if audio_buffer.size == 0:
            return "", {}

        try:
            # Flatten to 1D if needed (faster-whisper expects 1D float32 array at 16kHz)
            if audio_buffer.ndim > 1:
                audio_buffer = audio_buffer.flatten()

            # Calculate audio duration
            audio_duration = len(audio_buffer) / 16000  # 16kHz sample rate

            # Prepare context from previous transcriptions (token-limited)
            context_text = None
            context_tokens_count = len(self.context_tokens)

            if self.context_tokens:
                # Decode tokens directly (no re-encoding needed!)
                context_text = self.tokenizer.decode(self.context_tokens, skip_special_tokens=True)

            # Determine language hint: use last detected trusted language, or default
            lang = self.detected_language if self.detected_language else self.default_language
            if lang == "auto":
                lang = None

            segments, whisper_info = self.model.transcribe(
                audio_buffer,
                beam_size=5,
                language=lang,
                condition_on_previous_text=True,
                initial_prompt=context_text,
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=500
                )
            )

            # Collect transcription and update context with VAD timing info
            new_text_words = []
            segments_list = list(segments)  # Force evaluation of generator

            # Update detected language if it's a trusted one (en/fr)
            detected_lang = whisper_info.language
            if detected_lang in self.TRUSTED_LANGUAGES:
                self.detected_language = detected_lang

            # Build info dict for logging
            info = {
                'buffer_duration': audio_duration,
                'context_tokens_before': context_tokens_count,
                'has_speech': len(segments_list) > 0,
                'language': detected_lang,
                'language_probability': whisper_info.language_probability,
                'language_hint': lang  # What we asked for
            }

            if segments_list:
                first_segment = segments_list[0]
                last_segment = segments_list[-1]

                info.update({
                    'speech_start': first_segment.start,
                    'speech_end': last_segment.end,
                    'trimmed_start': first_segment.start,
                    'trimmed_end': audio_duration - last_segment.end,
                    'segments': []
                })

                for segment in segments_list:
                    info['segments'].append({
                        'start': segment.start,
                        'end': segment.end,
                        'text': segment.text
                    })
                    words = segment.text.strip().split()
                    new_text_words.extend(words)

            # Update context with new tokens (efficient: only tokenize new text once!)
            transcription = " ".join(new_text_words)
            if transcription:
                # Tokenize only the new text
                encoding = self.tokenizer.encode(transcription)
                new_tokens = encoding.ids if hasattr(encoding, 'ids') else encoding
                # Append new tokens to context
                self.context_tokens.extend(new_tokens)
                # Trim to max_context_tokens (keep most recent)
                if len(self.context_tokens) > self.max_context_tokens:
                    self.context_tokens = self.context_tokens[-self.max_context_tokens:]

            info['context_tokens_after'] = len(self.context_tokens)

            return transcription.strip(), info
        except Exception as e:
            print(f"GPU transcription failed: {e}")
            return "", {}

    def reset_context(self):
        """Clear context and detected language"""
        self.context_tokens = []
        self.detected_language = None
