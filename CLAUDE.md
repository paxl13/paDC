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
echo "claude-send" > /tmp/padc.fifo              # Send to Claude via claude-send-active script
echo "claude-send-fr" > /tmp/padc.fifo           # Send to Claude (French) via claude-send-active script
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

# Language code for GPU transcription (ISO 639-1 format)
PADC_LANGUAGE=en
# Default: en (English)
# Common options: en, fr, es, de, it, pt, ja, zh, etc.
# This controls the language for GPU (Faster Whisper) transcription

# Buffer size in seconds (rolling window of audio kept in memory)
PADC_BUFFER_SECONDS=30.0
# Default: 30 seconds
# Increase for longer continuous dictation without triggering transcription

# Audio gain normalization (0.0 to 1.0)
PADC_NORMALIZE_AUDIO=0.7
# Default: 0.7 (70% peak level, leaves headroom to prevent clipping)
# Set to 0.0 to disable and use raw microphone levels
# Recommended: 0.6-0.8 for most microphones

# Debug: Save audio buffers to WAV files for troubleshooting
PADC_DEBUG_SAVE_AUDIO=true
# When enabled, saves each buffer to debug_audio/ before transcription
# Useful for verifying rolling buffer behavior and diagnosing truncation issues
```

**Notes:**
- `PADC_ADAPTER` is not used in new_daemon.py (GPU-only, no adapter selection)
- The buffer is a rolling window - when it reaches the configured size, old audio is automatically discarded
- The buffer does NOT clear between transcriptions, ensuring no audio is lost

**Audio Normalization:**
- Automatically adjusts microphone gain to optimal levels for Whisper transcription
- Uses peak-based normalization: analyzes buffer, calculates required gain, applies safely
- Maximum amplification limited to 20dB (10x) to prevent over-amplification of very quiet audio
- Hard clips at ±1.0 to prevent overflow (rare with proper target levels)
- Gain information shown in transcription logs (e.g., "Gain: +12.5dB (peak: 15.3% → 70.0%)")
- Set `PADC_NORMALIZE_AUDIO=0.0` to disable and use raw mic levels (not recommended)

**Debug Audio Saving:**
- When `PADC_DEBUG_SAVE_AUDIO=true`, each audio buffer is saved to `debug_audio/buffer_YYYYMMDD_HHMMSS_ffffff.wav`
- Files are saved AFTER normalization, so you can verify what Whisper receives
- Files are saved in a background thread to avoid blocking transcription
- Useful for troubleshooting rolling buffer behavior, truncation issues, or VAD problems
- **WARNING:** Files accumulate quickly - remember to clean `debug_audio/` periodically

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
- `CLAUDE_SEND`: Send to Claude via `claude-send-active` script (doesn't touch clipboard)

**Key detail:** INSERT_CONTINUE restarts recording BEFORE transcription completes (line 361), creating seamless continuous dictation with no gap.

### Text Processing

The `process_text()` method (lines 313-329) fixes common Whisper mistakes:
- "cloud" / "Cloud" → "Claude"

Add new replacements to the `replacements` dict.

### Transcription Parameters (lines 368-378)

```python
self.model.transcribe(
    audio_buffer,
    language=self.language,             # Configurable via PADC_LANGUAGE (default: 'en')
    beam_size=5,
    condition_on_previous_text=True,    # Use prior segments
    initial_prompt=context_text,        # Context (token-limited)
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

### Adjusting Audio Gain/Normalization
The daemon automatically normalizes audio to optimal levels. To adjust:

1. **Change target level** in `.env`:
```bash
PADC_NORMALIZE_AUDIO=0.7  # 70% peak level (default)
# Try 0.6 for more headroom, 0.8 for louder audio
```

2. **Disable normalization** (use raw mic levels):
```bash
PADC_NORMALIZE_AUDIO=0.0
```

3. **Check gain in logs** - each transcription shows:
```
│ Gain: +12.5dB (peak: 15.3% → 70.0%)
```
- Positive dB = audio amplified
- Negative dB = audio attenuated
- "[CLIPPED]" warning = reduce target level

4. **Implementation details:**
- Normalization function: `normalize_audio_buffer()` (`daemon/new_daemon.py:50-110`)
- Uses peak-based normalization to prevent clipping
- Maximum gain limited to 20dB (10x) for safety
- Applied before transcription and before saving debug audio

### Adjusting Audio Feedback
- Start chime: `AudioRecorder.play_chime()` (lines 111-124)
- Cancel sound: `AudioRecorder.play_cancel_sound()` (lines 126-141)
- Frequency, duration, envelope parameters are configurable

### Debugging

#### General Debugging
```bash
# Run in foreground to see all logs
./a

# Or tail background logs
tail -f /tmp/padc_daemon.log
```

#### Debugging Audio Buffer Issues
To troubleshoot rolling buffer behavior, truncation issues, or VAD problems:

1. Enable debug audio saving in `.env`:
```bash
PADC_DEBUG_SAVE_AUDIO=true
```

2. Restart the daemon:
```bash
./a
```

3. Trigger transcriptions and check saved buffers:
```bash
# Buffers saved to debug_audio/ with timestamps
ls -lh debug_audio/
# Example: buffer_20251019_143022_123456.wav

# Play back to verify audio content
ffplay debug_audio/buffer_20251019_143022_123456.wav
# or
aplay debug_audio/buffer_20251019_143022_123456.wav
```

4. Analyze buffer properties:
```bash
# Check duration and sample rate
ffprobe debug_audio/buffer_20251019_143022_123456.wav
```

5. Clean up when done:
```bash
rm -rf debug_audio/
```

**Implementation details:**
- Saving happens in a background thread (`daemon/new_daemon.py:50-82`)
- WAV files are 16kHz mono, int16 format
- Files named with microsecond precision timestamps
- Buffer is saved BEFORE transcription, so you can verify what Whisper receives

### GPU Troubleshooting

If daemon exits with "ERROR: CUDA not available":
1. Check NVIDIA GPU: `nvidia-smi`
2. Verify CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`
3. Install CUDA/cuDNN if needed: `uv add torch`
4. Check library path in `./a` script points to correct venv Python version

## Insertion Methods

### xdotool Integration
Paste modes require `xdotool` (Linux X11 only):
- `INSERT` mode: `xdotool key shift+Insert`
- Clipboard saved/restored with 500ms delay to prevent race conditions

### tmux Integration
When a marked pane exists, text is sent directly via:
- `tmux send-keys -t {marked} "text "`
- Bypasses clipboard entirely

### Claude Send Integration
`CLAUDE_SEND` mode calls external script:
- `claude-send-active <text>`
- Doesn't touch clipboard
- Script must be in PATH
- Use for custom integrations (e.g., window manager bindings)

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
