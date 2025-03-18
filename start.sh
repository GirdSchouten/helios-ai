#!/bin/bash

# Navigate to the project directory
cd "$(dirname "$0")"

# Check if the virtual environment exists
if [ ! -d "helios_env" ]; then
    echo "Virtual environment not found! Creating one..."
    python3 -m venv helios_env
    source helios_env/bin/activate
    echo "Installing dependencies..."
    pip install -r requirements.txt
else
    echo "Activating virtual environment..."
    source helios_env/bin/activate
fi

# Run the AI governance script
echo "Starting Helios AI Governance..."
python helios_ai.py
