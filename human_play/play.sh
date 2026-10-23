#!/usr/bin/env bash
# ==============================================================================
# SNN-Plays-Doom: Interactive Human Play Launcher
# ==============================================================================
# Launches ViZDoom in interactive human mode with keyboard & mouse controls.
# Automatically detects the virtual environment and graphical display.
#
# Examples:
#   ./human_play/play.sh                        # Play Combat Arena (basic)
#   ./human_play/play.sh --scenario my_way_home # Play 3D Labyrinth Traversal
#   ./human_play/play.sh --scenario defend_the_center
#   ./human_play/play.sh --res 1024x768         # High-res mode
#   ./human_play/play.sh --sound                # Enable audio
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# Find python interpreter with vizdoom installed
VENV_PYTHON=""

if [ -n "$VIRTUAL_ENV" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
    VENV_PYTHON="$VIRTUAL_ENV/bin/python"
elif [ -x "/home/dilith_s_b_s/UoP/uop-venv/bin/python" ]; then
    VENV_PYTHON="/home/dilith_s_b_s/UoP/uop-venv/bin/python"
elif [ -x "$REPO_DIR/../../uop-venv/bin/python" ]; then
    VENV_PYTHON="$REPO_DIR/../../uop-venv/bin/python"
elif [ -x "$REPO_DIR/venv/bin/python" ]; then
    VENV_PYTHON="$REPO_DIR/venv/bin/python"
else
    VENV_PYTHON="$(which python3)"
fi

# Ensure DISPLAY is set for WSL2/Linux GUI window
if [ -z "$DISPLAY" ]; then
    export DISPLAY=:0
fi

echo "[Launcher] Using Python: $VENV_PYTHON"
echo "[Launcher] Display: $DISPLAY"
echo ""

exec "$VENV_PYTHON" "$SCRIPT_DIR/play.py" "$@"
