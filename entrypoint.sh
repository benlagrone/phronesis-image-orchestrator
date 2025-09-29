#!/usr/bin/env bash
set -euo pipefail

convert_checkpoint() {
  local src="$1"
  local dest="$2"
  local profile="$3"

  if [[ ! -f "$src" ]]; then
    echo "[entrypoint] checkpoint not found, skipping: $src"
    return
  fi

  if [[ -f "$dest/model_index.json" ]]; then
    echo "[entrypoint] Diffusers pipeline already present: $dest"
    return
  fi

  rm -rf "$dest"
  mkdir -p "$dest"

  echo "[entrypoint] Converting $src -> $dest via AutoPipeline"
  if python - "$src" "$dest" <<'PY'
import sys
from pathlib import Path
import torch
from diffusers import AutoPipelineForText2Image

src = Path(sys.argv[1])
dest = Path(sys.argv[2])

try:
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    pipe = AutoPipelineForText2Image.from_single_file(src.as_posix(), torch_dtype=dtype)
    pipe.save_pretrained(dest.as_posix())
    pipe = None
    print(f"[entrypoint] Saved pipeline to {dest}")
except Exception as exc:
    import traceback
    traceback.print_exc()
    sys.exit(1)
PY
  then
    echo "[entrypoint] Conversion complete: $dest"
  else
    echo "[entrypoint] Conversion failed for $src" >&2
  fi
}

echo "[entrypoint] Preparing Diffusers pipelines"
convert_checkpoint \
  "/models/Stable-diffusion/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors" \
  "/models/Stable-diffusion/Juggernaut-XL_v9-diffusers" \
  sdxl

convert_checkpoint \
  "/models/Stable-diffusion/Realistic_Vision_V6.safetensors" \
  "/models/Stable-diffusion/Realistic_Vision_v6-diffusers" \
  sd15

convert_checkpoint \
  "/models/Stable-diffusion/AbsoluteReality_v1.8.1.safetensors" \
  "/models/Stable-diffusion/AbsoluteReality_v1.8.1-diffusers" \
  sd15

LOG_LEVEL=${LOG_LEVEL:-info}
export PYTHONUNBUFFERED=1
echo "Starting uvicorn on port ${PORT:-8000} with log-level ${LOG_LEVEL}"
uvicorn app:app \
  --host 0.0.0.0 \
  --port ${PORT:-8000} \
  --log-level ${LOG_LEVEL} \
  --access-log
