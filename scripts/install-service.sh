#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SERVICE_NAME="cc-stt-daemon"

echo -e "${GREEN}=== CC-STT Daemon Service Installation ===${NC}"

# Check if user is in audio group
if ! groups | grep -q "\baudio\b"; then
    echo -e "${YELLOW}Warning: You are not in the 'audio' group.${NC}"
    echo "Run: sudo usermod -a -G audio $USER"
    echo "Then log out and back in for changes to take effect."
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create systemd user service directory
mkdir -p "$HOME/.config/systemd/user"

# Copy service file
SERVICE_SOURCE="$PROJECT_DIR/systemd/$SERVICE_NAME.service"
SERVICE_DEST="$HOME/.config/systemd/user/$SERVICE_NAME.service"

echo "Installing service file to $SERVICE_DEST"
sed "s|/home/jiang/cc/cc-stt|$PROJECT_DIR|g" "$SERVICE_SOURCE" > "$SERVICE_DEST"

# Set correct virtualenv path in service file
VENV_PATH="$PROJECT_DIR/.venv/bin/cc-stt-daemon"
sed -i "s|ExecStart=.*|ExecStart=$VENV_PATH|g" "$SERVICE_DEST"

# Reload systemd daemon
echo "Reloading systemd daemon"
systemctl --user daemon-reload

# Enable service
echo "Enabling $SERVICE_NAME service"
systemctl --user enable "$SERVICE_NAME"

echo -e "${GREEN}=== Installation Complete ===${NC}"
echo ""
echo "Available commands:"
echo "  systemctl --user start $SERVICE_NAME    # Start service"
echo "  systemctl --user stop $SERVICE_NAME     # Stop service"
echo "  systemctl --user restart $SERVICE_NAME  # Restart service"
echo "  systemctl --user status $SERVICE_NAME   # Check status"
echo "  journalctl --user -u $SERVICE_NAME -f   # View logs"
echo ""
echo -e "${YELLOW}Note: Make sure DISPLAY=:0 is set for GUI editor window${NC}"
