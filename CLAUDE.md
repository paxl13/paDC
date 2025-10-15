# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**paDC** is a GPU-only speech-to-text daemon for continuous dictation. It uses Faster Whisper with contextual transcription and buffer-based processing for maximum performance.

## ⚠️ Important: Current Implementation

**ONLY `daemon/new_daemon.py` is actively used.** All other code (`padc/`, `scripts/`, old daemon files) is deprecated legacy code kept for reference.

The active daemon is:
- **GPU-only** - exits with error if CUDA unavailable (no CPU fallback)
- **Single-file** - all functionality inlined in `daemon/new_daemon.py`
- **Buffer-based** - direct numpy array transcription (no temp files)
- **Contextual** - maintains 200-word rolling context across recordings

## Running the Daemon

### Primary Method: ./a Script

```bash
# Start daemon in foreground (recommended for development/debugging)
./a
```

This script:
- Sets up CUDA library paths automatically
- Runs `daemon/new_daemon.py --foreground` via uv
- Shows all output in terminal

### Background Mode

```bash
# Start as background daemon
uv run python daemon/new_daemon.py
# Logs go to /tmp/padc_daemon.log
```

### Send Commands via FIFO

```bash
# Available commands:
echo "insert" > /tmp/padc.fifo                    # Stop recording and paste once
echo "toggle-insert" > /tmp/padc.fifo            # Toggle with insert mode
echo "toggle-insert-continue" > /tmp/padc.fifo   # Toggle continuous mode
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

# Alternative variable name (both work)
WHISPER_MODEL=base
```

**Note:** `PADC_ADAPTER` is not used in new_daemon.py (GPU-only, no adapter selection).

## Architecture: daemon/new_daemon.py

### Core Classes

**AudioRecorder (lines 42-123)**
- Records at 16kHz mono using sounddevice
- Queue-based callback for real-time capture
- Audio feedback:
  - `play_chime()`: A440 sine wave (150ms) when recording starts
  - `play_cancel_sound()`: Descending E4→A3 tone when cancelled
- Returns numpy float32 arrays directly (no file writes)

**GPUWhisperModel (lines 129-246)**
- Wraps faster-whisper with GPU-only initialization
- CUDA verification: Creates test tensor to ensure GPU works (lines 141-152)
- Compute type fallback: Tries `int8` first, then `float16`
- **Exits on failure** - no CPU fallback (lines 159-194)
- **Contextual transcription**:
  - Maintains last 200 words as context (configurable)
  - Passes context as `initial_prompt` to Whisper
  - Auto-resets after 1 minute of inactivity (lines 280-292)
  - Manual reset via `reset-context` command

**PaDCDaemon (lines 252-577)**
- State machine: `IDLE` / `RECORDING` / `is_processing`
- FIFO-based command processing (`/tmp/padc.fifo`)
- Status file updates for tmux integration (`~/.padc_status`)
- Background transcription in separate thread (line 377)

### Recording Modes

- `NORMAL`: Copy to clipboard only
- `INSERT`: Paste with Shift+Insert (one-shot)
- `INSERT_CONTINUE`: Paste with Shift+Insert, then **auto-restart recording** for continuous dictation

**Key detail:** INSERT_CONTINUE restarts recording BEFORE transcription completes (line 361), creating seamless continuous dictation with no gap.

### Text Processing

The `process_text()` method (lines 313-329) fixes common Whisper mistakes:
- "cloud" / "Cloud" → "Claude"

Add new replacements to the `replacements` dict.

### Transcription Parameters (lines 216-226)

```python
self.model.transcribe(
    audio_buffer,
    language='en',                      # English only
    beam_size=5,
    condition_on_previous_text=True,    # Use prior segments
    initial_prompt=context_text,        # 200-word context
    vad_filter=True,                    # Voice activity detection
    vad_parameters=dict(
        min_silence_duration_ms=500     # 500ms silence detection
    )
)
```

### File Paths

- FIFO: `/tmp/padc.fifo` (command input)
- PID: `/tmp/padc_daemon.pid`
- Log: `/tmp/padc_daemon.log` (background mode only)
- Status: `~/.padc_status` (tmux status bar format)

## Common Development Tasks

### Adding Word Replacements
Edit `daemon/new_daemon.py:318-322`:
```python
replacements = {
    "cloud": "Claude",
    "Cloud": "Claude",
    # Add more here
}
```

### Changing Context Window
Modify `max_context_words` in GPUWhisperModel init (line 132):
```python
def __init__(self, model_size: str = "base", max_context_words: int = 200):
```

### Adjusting Audio Feedback
- Start chime: `AudioRecorder.play_chime()` (lines 59-72)
- Cancel sound: `AudioRecorder.play_cancel_sound()` (lines 74-89)
- Frequency, duration, envelope parameters are configurable

### Debugging
```bash
# Run in foreground to see all logs
./a

# Or tail background logs
tail -f /tmp/padc_daemon.log
```

### GPU Troubleshooting

If daemon exits with "ERROR: CUDA not available":
1. Check NVIDIA GPU: `nvidia-smi`
2. Verify CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`
3. Install CUDA/cuDNN if needed: `uv add torch`
4. Check library path in `./a` script points to correct venv Python version

## xdotool Integration

Paste modes require `xdotool` (Linux X11 only):
- `INSERT` mode: `xdotool key shift+Insert`
- Clipboard saved/restored with 500ms delay (lines 433, 439) to prevent race conditions

## Implementation Notes

### Threading Model
- Main thread: FIFO listener, command processing
- Audio thread: Sounddevice callback, queue processing
- Chime threads: Non-blocking audio feedback
- Transcription thread: Background GPU processing (line 377)

### Context Management
- Context persists across recordings until:
  - Manual reset: `echo "reset-context" > /tmp/padc.fifo`
  - Auto-reset: 1 minute of inactivity
  - Daemon shutdown
- Timer cancelled when recording starts (line 337)
- Timer started when entering idle state (line 367)

### Clipboard Behavior
- Always copies transcription to clipboard
- In INSERT modes: saves old clipboard, pastes, restores after 500ms
- In NORMAL mode: only copies, doesn't paste

### Model Loading
- Loads once at daemon startup (lines 272-278)
- No hot reloading - restart daemon to change models
- Model size controlled by `PADC_MODEL` or `WHISPER_MODEL` env var

## Legacy Code (Deprecated)

The following directories contain old code NOT used in production:
- `padc/`: Original CLI implementation with adapter pattern
- `padc/adapters/`: CPU/GPU/OpenAI adapters (modular design)
- `daemon/padc_daemon.py`: Old daemon with adapter support
- `scripts/`: Helper scripts for old daemon
- `padc-gpu`: Wrapper for old GPU adapter

These files are kept for reference but should not be modified or used.
