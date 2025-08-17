FROM python:3.10-slim

WORKDIR /app

# Install Python dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY entrypoint.sh /app/entrypoint.sh
COPY app.py /app/app.py

RUN pip install --no-cache-dir fastapi uvicorn[standard] diffusers torch torchvision safetensors pillow

RUN chmod +x /app/entrypoint.sh

VOLUME ["/models", "/output"]

EXPOSE 7860

ENTRYPOINT ["/app/entrypoint.sh"]
