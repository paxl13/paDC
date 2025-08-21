# paDC - Speech to Text CLI

A command-line tool for speech-to-text transcription with support for local (CPU/GPU) and cloud-based transcription.

## Features

- **Multiple Recording Modes**:
  - Daemon mode: Background recording with start/stop commands
  - Live mode: Interactive recording with Enter key control  
  - Single session: One-time recording
  - Toggle: Smart start/stop based on daemon state

- **Multiple STT Adapters**:
  - `local`: CPU-based Faster Whisper (int8 quantization)
  - `local_gpu`: GPU-based Faster Whisper (requires CUDA)
  - `openai`: OpenAI Whisper API (requires API key)

- **Output Options**:
  - Clipboard: Automatically copies transcription to clipboard
  - Paste: Types the text using xdotool (`--paste`)
  - Insert: Pastes using Shift+Insert (`--insert`)

## Installation

```bash
# Install with uv (editable mode for development)
uv tool install --editable .

# Or install globally
uv tool install .
```

## Configuration

Copy `.env.example` to `.env` and configure:

```bash
# Default adapter (local, local_gpu, or openai)
PADC_ADAPTER=local

# Model size for local adapters (tiny, base, small, medium, large)
WHISPER_MODEL=base

# OpenAI API key (for openai adapter)
OPENAI_API_KEY=your_key_here
```

## Usage

```bash
# Single recording (default)
padc

# Daemon mode
padc start              # Start recording
padc stop               # Stop and transcribe
padc stop --paste       # Stop and type the text
padc stop --insert      # Stop and paste with Shift+Insert

# Toggle daemon
padc toggle             # Start if not running, stop if running
padc toggle --paste     # With paste option when stopping

# Live mode (interactive)
padc live               # Press Enter to start/stop recording

# Specify adapter
padc record --adapter local_gpu
padc start --adapter openai
```

## GPU Support Troubleshooting

If you encounter CUDA/cuDNN errors when using `local_gpu`:

### Missing CUDA Libraries Error
```
Unable to load any of {libcudnn_ops.so.9.1.0, libcudnn_ops.so.9.1, ...}
```

**Solutions:**

1. **Install CUDA and cuDNN** (if you have an NVIDIA GPU):
   ```bash
   # Ubuntu/Debian
   sudo apt install nvidia-cuda-toolkit libcudnn8
   
   # Or use conda/mamba
   conda install cudatoolkit cudnn
   ```

2. **Install PyTorch with CUDA support** (optional, for better detection):
   ```bash
   uv add torch
   ```

3. **Use CPU adapter instead** (if no GPU available):
   ```bash
   padc --adapter local
   ```

The `local_gpu` adapter will automatically fall back to CPU if:
- CUDA is not available
- GPU initialization fails
- Runtime GPU errors occur

### Performance Notes

- `local`: Best for CPU-only systems (uses int8 quantization)
- `local_gpu`: Best for NVIDIA GPU systems (uses int8 or float16)
- Both adapters will use the model size specified in `WHISPER_MODEL`
- Larger models are more accurate but slower

## Requirements

- Python 3.12+
- For `local_gpu`: NVIDIA GPU with CUDA support
- For paste features: `xdotool` (Linux/X11)

## License

MIT