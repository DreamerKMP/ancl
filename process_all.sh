#!/bin/bash

# Default directories
INPUT_DIR=${1:-"tg_movies"}
OUTPUT_DIR=${2:-"analyzed_movies"}

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Ensure virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    ./venv/bin/pip install opencv-python-headless mediapipe numpy
fi

echo "Processing videos from $INPUT_DIR to $OUTPUT_DIR..."

# The python script now handles the loop internally via --input-dir
./venv/bin/python3 analyze_bouldering.py --input-dir "$INPUT_DIR" --output-dir "$OUTPUT_DIR"

echo "All videos processed. Check '$OUTPUT_DIR' folder."
