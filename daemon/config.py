"""Configuration centralisee pour paDC daemon"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


# Paths
FIFO_PATH = Path("/tmp/padc.fifo")
PID_FILE = Path("/tmp/padc_daemon.pid")
STATUS_FILE = Path.home() / ".padc_status"

# Get project root (where the daemon package is located)
PROJECT_ROOT = Path(__file__).parent.parent
TRANSCRIPTION_LOG = PROJECT_ROOT / "transcriptions.log"
DEBUG_AUDIO_DIR = PROJECT_ROOT / "debug_audio"


@dataclass
class AudioConfig:
    """Configuration audio"""
    sample_rate: int = 16000
    channels: int = 1
    buffer_seconds: float = 30.0
    normalize_target: float = 0.7
    debug_save_audio: bool = False


@dataclass
class WhisperConfig:
    """Configuration du modele Whisper"""
    model_size: str = "base"
    language: str = "en"
    max_context_tokens: int = 200


@dataclass
class RealtimeConfig:
    """Configuration du mode temps reel"""
    silence_threshold: float = 5.0
    max_interval: float = 20.0
    rms_threshold: float = 0.01


@dataclass
class HardwareGainConfig:
    """Configuration du controle de gain materiel"""
    enabled: bool = True
    min_gain: float = 20.0
    max_gain: float = 150.0
    target_peak: float = 0.5


@dataclass
class DaemonConfig:
    """Configuration complete du daemon"""
    audio: AudioConfig = field(default_factory=AudioConfig)
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    realtime: RealtimeConfig = field(default_factory=RealtimeConfig)
    hw_gain: HardwareGainConfig = field(default_factory=HardwareGainConfig)

    @classmethod
    def from_env(cls) -> "DaemonConfig":
        """Charge la configuration depuis les variables d'environnement"""
        return cls(
            audio=AudioConfig(
                buffer_seconds=float(os.environ.get("PADC_BUFFER_SECONDS", "30.0")),
                normalize_target=float(os.environ.get("PADC_NORMALIZE_AUDIO", "0.7")),
                debug_save_audio=os.environ.get("PADC_DEBUG_SAVE_AUDIO", "false").lower() == "true",
            ),
            whisper=WhisperConfig(
                model_size=os.environ.get("PADC_MODEL", os.environ.get("WHISPER_MODEL", "base")),
                language=os.environ.get("PADC_LANGUAGE", "en"),
            ),
            realtime=RealtimeConfig(
                silence_threshold=float(os.environ.get("PADC_REALTIME_SILENCE_THRESHOLD", "5.0")),
                max_interval=float(os.environ.get("PADC_REALTIME_MAX_INTERVAL", "20.0")),
                rms_threshold=float(os.environ.get("PADC_SILENCE_RMS_THRESHOLD", "0.01")),
            ),
            hw_gain=HardwareGainConfig(
                enabled=os.environ.get("PADC_HW_GAIN_CONTROL", "true").lower() == "true",
                min_gain=float(os.environ.get("PADC_HW_GAIN_MIN", "20")),
                max_gain=float(os.environ.get("PADC_HW_GAIN_MAX", "150")),
                target_peak=float(os.environ.get("PADC_HW_GAIN_TARGET", "0.5")),
            ),
        )
