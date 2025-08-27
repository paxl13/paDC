#!/bin/bash
# Install PADC with ultra-fast FIFO architecture

set -e

# Default installation directory
DEFAULT_INSTALL_DIR="$HOME/bin/tools"

# Parse command line arguments
INSTALL_DIR="${1:-$DEFAULT_INSTALL_DIR}"

echo "Installing PADC to $INSTALL_DIR..."

# Get script directory (absolute path)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Create installation directory if it doesn't exist
mkdir -p "$INSTALL_DIR"

# Create symlinks to the actual scripts
ln -sf "$SCRIPT_DIR/padc" "$INSTALL_DIR/padc"
ln -sf "$SCRIPT_DIR/padcd" "$INSTALL_DIR/padcd"

# Check if install directory is in PATH
if [[ ":$PATH:" != *":$INSTALL_DIR:"* ]]; then
    echo ""
    echo "⚠️  Warning: $INSTALL_DIR is not in your PATH"
    echo "Add the following line to your shell configuration file (~/.bashrc, ~/.zshrc, etc.):"
    echo ""
    echo "  export PATH=\"\$PATH:$INSTALL_DIR\""
    echo ""
fi

echo "Installation complete!"
echo ""
echo "Usage:"
echo "  padc toggle         # Toggle recording (copy to clipboard)"
echo "  padc toggle-paste   # Toggle recording and type text"
echo "  padc toggle-insert  # Toggle recording and paste with Shift+Insert"
echo "  padc start          # Start recording"
echo "  padc stop           # Stop recording"
echo "  padc status         # Check daemon status"
echo ""
echo "Daemon management:"
echo "  padcd start         # Start the daemon"
echo "  padcd stop          # Stop the daemon"
echo "  padcd restart       # Restart the daemon"
echo "  padcd status        # Check daemon status"
echo "  padcd logs [-f]     # View daemon logs"
echo ""
echo "The daemon will auto-start on first use."
echo "Model loading happens once at daemon startup (~5-10s)."
echo "After that, all commands are instant (<2ms)."