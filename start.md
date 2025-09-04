Here is a detailed start.md file you can include at the root of your repository. This README is tailored for use with GitHub Copilot/Codex and assumes you’re setting up a scalable Docker-based Stable Diffusion API service.

⸻


# 🚀 Stable Diffusion API Service with Auto-Scaling Support

This repository sets up a **Dockerized Stable Diffusion API service** optimized for:

- ✅ Real-time image generation from text prompts
- 🐾 Animal portraits, 🕊️ biblical scenes, 🏠 real estate marketing stills
- ⚙️ FastAPI-compatible API
- 📦 Supports SDXL, LoRAs, ESRGAN upscaling
- 📈 Spin up multiple containers for concurrent video generation workflows
- 🛑 Graceful shutdown when containers are idle or complete

---

## 📁 Project Structure

stable-diffusion-api/
├── Dockerfile
├── docker-compose.yml
├── entrypoint.sh
├── start.md        <– You are here
├── models/         <– Mount your SDXL, LoRA, ESRGAN, etc.
└── output/         <– Rendered images will be stored here

---

## 🔧 Prerequisites

- ✅ Ubuntu 20.04+ (already installed)
- ✅ Docker + Docker Compose
- ✅ NVIDIA drivers + CUDA Toolkit
- ✅ Models downloaded to `./models` directory
  - Examples:
    - `models/Stable-diffusion/sd_xl_base_1.0.safetensors`
    - `models/Lora/real_estate_style.safetensors`
    - `models/ESRGAN/R-ESRGAN-4x.pth`

---

## 🛠️ Setup Instructions

### 1. Clone This Repository

```bash
git clone https://github.com/your-username/stable-diffusion-api.git
cd stable-diffusion-api

2. Prepare Volumes

Place your models in:

mkdir -p models/Stable-diffusion
mkdir -p output

Then move your .safetensors and .ckpt files into the appropriate folders.

⸻

3. Build the Docker Image

```bash
docker compose build
```


⸻

4. Run the Service

Option A: Run One Instance

```bash
docker compose up -d
```

> **Tip:** Older installations using `docker-compose` may throw `KeyError: 'ContainerConfig'` during build.
> Disable BuildKit to work around it:
>
> ```bash
> export DOCKER_BUILDKIT=0 COMPOSE_DOCKER_CLI_BUILD=0
> docker-compose up -d
> ```


Service will be available at (default port **8000**, configurable via `PORT` env variable):
➡️ http://localhost:8000/docs (FastAPI docs)
➡️ POST `/txt2img` to generate an image
➡️ GET `/files/{filename}` to download generated images

Option B: Scale Multiple Instances (optional)

```bash
docker compose up -d --scale sd-api=3
```

Notes:
- Replace `3` with any integer (e.g., `1`, `2`, `4`).
- Do not use a literal `N`. If you want a variable: `COUNT=3 docker compose up -d --scale sd-api=$COUNT`.
- You can scale up/down later with the same command.

⸻

## 🔧 Environment (.env)

Create a `.env` file in the repo root to point the service at your models and output directories. Use absolute paths (Compose does not expand `~`).

Examples:

MacOS:

```
SD_MODELS_DIR=/Users/yourname/sd-models
SD_OUTPUT_DIR=/Users/yourname/sd-outputs  # optional (defaults to ./output)
LOG_LEVEL=debug                           # optional
```

Ubuntu:

```
SD_MODELS_DIR=/home/youruser/sd-models
SD_OUTPUT_DIR=/home/youruser/sd-outputs   # optional (defaults to ./output)
LOG_LEVEL=debug                           # optional
```

Notes:
- Ensure `SD_MODELS_DIR` exists on the same host where you run `docker compose`.
- The folder must contain `Stable-diffusion/` with your `.safetensors`/`.ckpt` files.

## 📸 Output Directory (Pinned)

This project uses the container path `/output` for generated images. Docker Compose maps it via `SD_OUTPUT_DIR`.

- Pinned output location (do not change unless explicitly requested):
  - Ubuntu host: `SD_OUTPUT_DIR=/home/master-benjamin/Pictures`
- Add to `.env` on the Ubuntu server:

```
SD_OUTPUT_DIR=/home/master-benjamin/Pictures
```

Access files at: `http://localhost:8000/files/<filename>`


## ▶️ Start + Logs

```
docker network create fortress-phronesis-net || true
docker compose up -d --build
docker compose logs -f --tail=100 sd-api
```

Health check:

```
curl -s http://127.0.0.1:8000/health
```

## 📟 Watching Logs

Common ways to view logs for the `sd-api` service:

```
# Follow live logs (recommended)
docker compose logs -f --tail=100 sd-api

# Show recent logs without following
docker compose logs --tail=200 sd-api

# Logs since a time window
docker compose logs --since=10m sd-api

# Run in foreground (prints logs to the terminal until Ctrl+C)
docker compose up --build

# If needed, get the container name and use docker logs directly
docker compose ps
docker logs -f <container_name>
```

## 🟢 Ubuntu Setup Commands (Copy/Paste)

Run these on the Ubuntu host where you execute `docker compose`.

```
# 1) Create shared models dir (if needed)
mkdir -p /home/master-benjamin/sd-models

# 2) Create .env with pinned paths and verbose logging
cat > .env << 'EOF'
SD_MODELS_DIR=/home/master-benjamin/sd-models
SD_OUTPUT_DIR=/home/master-benjamin/Pictures
LOG_LEVEL=debug
EOF

# 3) Bring up the stack and follow logs
docker network create fortress-phronesis-net || true
docker compose up -d --build
docker compose logs -f --tail=100 sd-api

# 4) Health check
curl -s http://127.0.0.1:8000/health
```

## 🧪 Example API Call (JSON)

Use the JSON endpoint compatible with A1111-style payloads.

```bash
curl --location 'http://127.0.0.1:8000/sdapi/v1/txt2img' \
  --header 'Content-Type: application/json' \
  --data '{
    "model": "Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors",
    "height": 720,
    "width": 1280,
    "steps": 20,
    "cfg_scale": 8.5,
    "seed": 931990,
    "batch_size": 1,
    "batch_count": 1,
    "prompt": "a rustic house at sunset",
    "negative_prompt": "blurry, noisy"
  }'
```

Response:

```json
{
  "ok": true,
  "count": 1,
  "paths": ["/output/out-<id>.png"],
  "urls": ["http://localhost:8000/files/out-<id>.png"]
}
```

You can then download with:

```bash
wget http://localhost:8000/files/out-<id>.png -O out.png
```

All images are also saved to the host-mounted `./output` directory.

⸻

🔄 Stopping & Scaling Down

To shut down all containers:

```bash
docker compose down
```

To scale up/down:

```bash
docker compose up -d --scale sd-api=3
```

Replace `3` with the desired number of replicas.

To monitor resource usage:

docker stats


⸻

🚦 Optional: Auto-Scaling with Python

Use the Docker SDK to programmatically scale containers:

import docker
client = docker.from_env()
service = client.services.get("sd-api")
service.scale(5)  # spin up 5 containers


⸻

📌 TODO / Roadmap
	•	Add NGINX gateway for load-balanced access to multiple API instances
	•	GPU load detection for auto-scaling based on video pipeline workload
	•	Healthcheck endpoint for container lifecycle management
	•	Mount logging directory for monitoring/stats
	•	Add authentication token for secured inference API

⸻

📜 License

MIT License.

⸻

✝️ Bless this build to serve your vision with clarity and beauty.
