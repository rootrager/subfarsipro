#!/bin/bash
# Script to run SubFarsiPro with proper virtualenv activation

# Path to virtualenv
VENV_PATH="/home/miggor/programming/myenv"

# Check if virtualenv exists
if [ ! -d "$VENV_PATH" ]; then
    echo "❌ Virtualenv not found at: $VENV_PATH"
    echo "Please update VENV_PATH in this script or activate your virtualenv manually"
    exit 1
fi

# Activate virtualenv
source "$VENV_PATH/bin/activate"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run the program
python3 "$SCRIPT_DIR/subfarsipro/subfarsipro_v3.py" "$@"

# Deactivate virtualenv (optional)
deactivate 2>/dev/null || true

