#!/bin/bash

set -e # Exit on error

if [ -d "venv" ]; then
    echo "Virtual environment already exists. Skipping setup."
else
    echo "Creating virtual environment..."
    python3 -m venv venv

    source venv/bin/activate

    pip install --upgrade pip

    if [ -f requirements.txt ]; then
        echo "Installing dependencies from requirements.txt..."
        pip install -r requirements.txt
    else
        echo "No requirements.txt found. Skipping dependency installation."
    fi

    echo "Setup complete."
fi

# Activate and run script
source venv/bin/activate

echo "Running rag_app.py..."

streamlit run rag_app.py
