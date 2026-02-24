#!/bin/bash

# Check arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_directory> <output_directory>"
    exit 1
fi

INPUT_DIR=$(realpath "$1")
OUTPUT_DIR=$(realpath "$2")

# Build the container image
echo "Building Podman image..."
podman build -t bouldering-analyser .

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run the container
echo "Running analysis on videos in $INPUT_DIR..."

podman run --rm \
    -v "$INPUT_DIR":/app/input:Z \
    -v "$OUTPUT_DIR":/app/output:Z \
    bouldering-analyser --input-dir /app/input --output-dir /app/output

echo "Done. Results are in '$OUTPUT_DIR' folder."
