FROM docker.io/library/python:3.12-slim

# Install system dependencies for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir \
    opencv-python-headless \
    mediapipe \
    numpy

# Copy the script
COPY analyze_bouldering.py .

# Pre-download the model using wget (often more robust with SSL in containers)
RUN wget --no-check-certificate https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task

# Entry point
ENTRYPOINT ["python3", "analyze_bouldering.py"]
