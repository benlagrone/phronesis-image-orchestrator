#!/usr/bin/env bash
set -euo pipefail

# Download Stable Diffusion checkpoints/pipelines listed below.
# Requires:
#   1) `hf` CLI on PATH (pip install --upgrade huggingface_hub)
#   2) `HF_TOKEN` environment variable with a read token
#   3) Enough disk space under $HOME/sd-models

if ! command -v hf >/dev/null 2>&1; then
  echo "[error] hf CLI not found. Add \"export PATH=\"\$HOME/.local/bin:\$PATH\"\" to your shell or install huggingface_hub." >&2
  exit 1
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "[error] HF_TOKEN is not set. Run: export HF_TOKEN=hf_xxx" >&2
  exit 1
fi

BASE_DIR="${MODEL_BASE_DIR:-$HOME/sd-models/Stable-diffusion}"
LOG_DIR="${MODEL_LOG_DIR:-$HOME/sd-models/logs}"
mkdir -p "$BASE_DIR" "$LOG_DIR"

ts="$(date +%Y%m%d-%H%M%S)"
LOG_FILE="$LOG_DIR/download-$ts.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[info] Logging to $LOG_FILE"

# Models to download as full Diffusers directories.
# WARNING: many sources below are community mirrors. Review before trusting in production.
declare -A DIFFUSERS_SOURCES=(
  [AbsoluteReality_v1.8.1]="digiplay/AbsoluteReality_v1.8.1"
  [Photon_v1]="digiplay/Photon_v1"
  [Realistic_Vision_v6]="Googh23/Realistic_Vision_v6"
)

# Models to download as single checkpoint files (filename@repo -> destination name).
declare -A CHECKPOINT_SOURCES=(
  [digiplay/AbsoluteReality_v1.8.1:absolutereality_v181.safetensors]="AbsoluteReality_v1.8.1.safetensors"
  [digiplay/Photon_v1:photon_v1.safetensors]="Photon_v1.safetensors"
  [Googh23/Realistic_Vision_v6:Realistic_Vision_V6.safetensors]="Realistic_Vision_V6.safetensors"
)

# checkpoint_dir|checkpoint_filename|profile (sd15 or sdxl)
CONVERSION_TARGETS=(
  "Juggernaut-XL_v9-diffusers|Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors|sdxl"
  "Realistic_Vision_v6-diffusers|Realistic_Vision_V6.safetensors|sd15"
  "AbsoluteReality_v1.8.1-diffusers|AbsoluteReality_v1.8.1.safetensors|sd15"
)

download_diffusers_repo() {
  local name="$1"
  local repo="$2"
  local target="$BASE_DIR/$name"
  echo "[info] === $name (Diffusers repo: $repo) ==="
  rm -rf "$target.tmp"
  mkdir -p "$target.tmp"
  if hf download "$repo" --repo-type model --include "*" --local-dir "$target.tmp"; then
    rm -rf "$target"
    mv "$target.tmp" "$target"
    echo "[info] -> stored at $target"
  else
    echo "[warn] download failed for $repo; leaving $target.tmp for inspection" >&2
  fi
}

download_checkpoint() {
  local spec="$1" dest="$2"
  local repo="${spec%%:*}"
  local file="${spec##*:}"
  local target="$BASE_DIR/$dest"
  echo "[info] === $dest (checkpoint: $repo/$file) ==="
  hf download "$repo" "$file" --repo-type model --local-dir "$BASE_DIR"
  echo "[info] -> stored at $target"
}

convert_checkpoint_to_diffusers() {
  local rel_dir="$1"
  local checkpoint_file="$2"
  local profile="$3"
  local abs_checkpoint="$BASE_DIR/$checkpoint_file"

  if [[ ! -f "$abs_checkpoint" ]]; then
    echo "[warn] Conversion skipped; checkpoint not found: $abs_checkpoint"
    return
  fi

  local dest_dir="/models/Stable-diffusion/$rel_dir"
  local convert_cmd
  case "$profile" in
    sdxl)
      convert_cmd="python -m diffusers.pipelines.stable_diffusion.convert_original_stable_diffusion_to_diffusers --checkpoint_path '$abs_checkpoint' --dump_path '$dest_dir' --from_safetensors --pipeline_class StableDiffusionXLPipeline"
      ;;
    sd15)
      convert_cmd="python -m diffusers.pipelines.stable_diffusion.convert_original_stable_diffusion_to_diffusers --checkpoint_path '$abs_checkpoint' --dump_path '$dest_dir' --from_safetensors"
      ;;
    *)
      echo "[warn] Unknown conversion profile '$profile' for $checkpoint_file" >&2
      return
      ;;
  esac

  echo "[info] Converting $checkpoint_file -> $dest_dir ($profile)"
  if docker compose exec -T sd-api bash -lc "$convert_cmd"; then
    echo "[info] Conversion finished: $dest_dir"
  else
    echo "[warn] Conversion failed for $checkpoint_file" >&2
  fi
}

echo "[info] Starting Diffusers repo downloads"
for name in "${!DIFFUSERS_SOURCES[@]}"; do
  download_diffusers_repo "$name" "${DIFFUSERS_SOURCES[$name]}"
done

echo "[info] Starting single checkpoint downloads"
for spec in "${!CHECKPOINT_SOURCES[@]}"; do
  download_checkpoint "$spec" "${CHECKPOINT_SOURCES[$spec]}"
done

echo "[info] Converting checkpoints to Diffusers format"
for entry in "${CONVERSION_TARGETS[@]}"; do
  IFS='|' read -r rel_dir checkpoint profile <<<"$entry"
  convert_checkpoint_to_diffusers "$rel_dir" "$checkpoint" "$profile"
done

echo "[info] All requested downloads attempted. Review $LOG_FILE for details."
