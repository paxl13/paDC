# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**paDC** is a GPU-only speech-to-text daemon for continuous dictation. It uses Faster Whisper with contextual transcription and buffer-based processing for maximum performance.

## Architecture

The daemon is organized into 4 modules:

```
daemon/
├── __init__.py       # Package marker
├── config.py         # Configuration centralisee (dataclasses)
├── audio.py          # AudioRecorder + chimes + normalizer + MicGainController
├── whisper.py        # GPUWhisperModel + context + text processing
└── main.py           # Entry point + FIFO + orchestration (PaDCDaemon)
```

### Module Responsibilities

| Module | Content |
|--------|---------|
| `config.py` | Dataclasses (`AudioConfig`, `WhisperConfig`, `RealtimeConfig`, `HardwareGainConfig`, `DaemonConfig`), `from_env()`, path constants |
| `audio.py` | `AudioRecorder`, `MicGainController`, chimes (`play_*`), `normalize_audio_buffer()`, `save_audio_buffer_to_wav()` |
| `whisper.py` | `GPUWhisperModel`, `is_hallucination()`, `remove_filler_words()`, `process_text()`, hallucination patterns |
| `main.py` | `PaDCDaemon` (orchestration), FIFO loop, insertion methods, `main()` |

## Running the Daemon

### Primary Method: ./a Script

```bash
# Start daemon in foreground (recommended for development/debugging)
./a
```

This script:
- Sets up CUDA library paths automatically
- Runs `python -m daemon.main` via uv
- Shows all output in terminal

### Background Mode

```bash
uv run python -m daemon.main
```

### Send Commands via FIFO

```bash
echo "insert" > /tmp/padc.fifo                    # Stop recording and paste once
echo "toggle-insert" > /tmp/padc.fifo            # Toggle with insert mode
echo "toggle-insert-continue" > /tmp/padc.fifo   # Toggle continuous mode
echo "claude-send" > /tmp/padc.fifo              # Send to Claude via claude-send-active script
echo "toggle-realtime" > /tmp/padc.fifo          # Toggle real-time mode
echo "toggle-realtime-insert" > /tmp/padc.fifo   # Toggle real-time mode (INSERT)
echo "cancel" > /tmp/padc.fifo                   # Cancel without transcribing
echo "reset-context" > /tmp/padc.fifo            # Clear transcription context
echo "shutdown" > /tmp/padc.fifo                 # Stop daemon
```

## Configuration

Create `.env` file (copy from `.env.example`):

```bash
# Model size - larger = more accurate but slower
PADC_MODEL=base
# Options: tiny, base, small, medium, large, large-v2, large-v3

# Language code for GPU transcription (ISO 639-1 format)
PADC_LANGUAGE=en

# Buffer size in seconds (rolling window of audio kept in memory)
PADC_BUFFER_SECONDS=30.0

# Audio gain normalization (0.0 to 1.0)
PADC_NORMALIZE_AUDIO=0.7

# Debug: Save audio buffers to WAV files for troubleshooting
PADC_DEBUG_SAVE_AUDIO=false

# Real-time mode configuration
PADC_REALTIME_SILENCE_THRESHOLD=5.0
PADC_REALTIME_MAX_INTERVAL=20.0
PADC_SILENCE_RMS_THRESHOLD=0.01

# Hardware gain control
PADC_HW_GAIN_CONTROL=true
PADC_HW_GAIN_MIN=20
PADC_HW_GAIN_MAX=150
PADC_HW_GAIN_TARGET=0.5
```

## Core Classes

### config.py

```python
@dataclass
class DaemonConfig:
    audio: AudioConfig
    whisper: WhisperConfig
    realtime: RealtimeConfig
    hw_gain: HardwareGainConfig

    @classmethod
    def from_env(cls) -> "DaemonConfig": ...
```

### audio.py

**AudioRecorder**
- Records at 16kHz mono using sounddevice
- Queue-based callback for real-time capture
- Audio feedback methods: `play_chime()`, `play_cancel_sound()`, `play_realtime_*_chime()`
- Returns numpy float32 arrays directly (no file writes)

**MicGainController**
- Adaptive hardware microphone gain control via PulseAudio/PipeWire
- Adjusts gain based on measured peak levels

**Helper Functions**
- `normalize_audio_buffer()`: Peak-based normalization
- `save_audio_buffer_to_wav()`: Debug audio saving (background thread)

### whisper.py

**GPUWhisperModel**
- Wraps faster-whisper with GPU-only initialization
- CUDA verification: Creates test tensor to ensure GPU works
- Compute type fallback: Tries `int8` first, then `float16`
- **Exits on failure** - no CPU fallback
- **Contextual transcription**: Maintains last 200 tokens as context

**Helper Functions**
- `is_hallucination()`: Filters known Whisper hallucination patterns
- `remove_filler_words()`: Removes "euh", "hmm", etc.
- `process_text()`: Word replacements (cloud -> Claude)

### main.py

**PaDCDaemon**
- State machine: `IDLE` / `RECORDING` / `is_processing`
- FIFO-based command processing (`/tmp/padc.fifo`)
- Status file updates for tmux integration (`~/.padc_status`)
- Background transcription in separate thread

**Recording Modes**
- `NORMAL`: Copy to clipboard only
- `INSERT`: Paste with Shift+Insert (one-shot)
- `INSERT_CONTINUE`: Paste with Shift+Insert, then auto-restart recording
- `CLAUDE_SEND`: Send to Claude via `claude-send-active` script

## Common Development Tasks

### Adding Word Replacements
Edit `daemon/whisper.py` in `process_text()`:
```python
replacements = {
    "cloud": "Claude",
    "Cloud": "Claude",
    # Add more here
}
```

### Adding Hallucination Patterns
Edit `daemon/whisper.py`:
```python
HALLUCINATION_PATTERNS = [
    re.compile(r"^Your pattern here.*", re.IGNORECASE),
    # ...
]
```

### Changing Context Window
Modify `max_context_tokens` in `daemon/config.py` `WhisperConfig`.

### Adjusting Audio Gain/Normalization
1. Change target level in `.env`: `PADC_NORMALIZE_AUDIO=0.7`
2. Disable normalization: `PADC_NORMALIZE_AUDIO=0.0`
3. Check gain in logs: `│ Gain: +12.5dB (peak: 15.3% → 70.0%)`

### Debugging

```bash
# Run in foreground to see all logs
./a

# Enable debug audio saving
PADC_DEBUG_SAVE_AUDIO=true ./a
```

### GPU Troubleshooting

If daemon exits with "ERROR: CUDA not available":
1. Check NVIDIA GPU: `nvidia-smi`
2. Verify CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`
3. Check library path in `./a` script points to correct venv Python version

## File Paths

- FIFO: `/tmp/padc.fifo` (command input)
- PID: `/tmp/padc_daemon.pid`
- Status: `~/.padc_status` (tmux status bar format)
- Transcription log: `transcriptions.log` (project root)
- Debug audio: `debug_audio/` (project root)

## Threading Model

- Main thread: FIFO listener, command processing
- Audio thread: Sounddevice callback, queue processing
- Chime threads: Non-blocking audio feedback
- Transcription thread: Background GPU processing
- Real-time monitor thread: Silence detection for auto-transcription
