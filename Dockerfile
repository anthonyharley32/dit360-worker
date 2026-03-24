# DiT360 Serverless Worker for RunPod
# Generates 360° panoramic images from text prompts

FROM runpod/base:0.6.2-cuda12.2.0

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Clone DiT360
RUN git clone https://github.com/Insta360-Research-Team/DiT360.git

# Install PyTorch and dependencies
WORKDIR /app/DiT360
RUN pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu121 && \
    pip install -r requirements.txt

# Pre-download the model from HuggingFace at build time (avoids cold start download)
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('Insta360-Research/DiT360-Panorama-Image-Generation', local_dir='./models/DiT360')"

# Copy handler
WORKDIR /app
COPY handler.py /app/handler.py

# Install runpod SDK
RUN pip install runpod

CMD ["python", "/app/handler.py"]
