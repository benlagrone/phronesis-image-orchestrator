# Recommended Stable Diffusion Models for Realistic People Generation

The following models are well regarded for generating photorealistic or stylised images of people. Each entry includes a short description and a download command that places the model into `~/sd-models/`.

> **Reminder:** Set your Hugging Face token in the shell before running any command:
>
> ```bash
> export HF_TOKEN="hf_your_read_token"
> export PATH="$HOME/.local/bin:$PATH"  # ensure the `hf` CLI is available
> ```

## RealVisXL V4.0 (SDXL)
- Balanced realism, responsive to prompts, good skin rendering.
- Download (checkpoint):
  ```bash
  wget --header="Authorization: Bearer $HF_TOKEN" \
    "https://huggingface.co/SG161222/RealVisXL_V4.0/resolve/main/RealVisXL_V4.0.safetensors" \
    -O "$HOME/sd-models/Stable-diffusion/RealVisXL_V4.0.safetensors"
  ```

## Photon v1 (community mirror, SDXL)
- Neutral-toned SDXL checkpoint optimised for lifelike portraits.
- Download full Diffusers pipeline:
  ```bash
  mkdir -p "$HOME/sd-models/Stable-diffusion/Photon_v1"
  hf download digiplay/Photon_v1 --repo-type model --include "*" \
    --local-dir "$HOME/sd-models/Stable-diffusion/Photon_v1"
  ```

## AbsoluteReality v1.8.1 (community mirror, SD 1.5)
- Reliable v1.5 model with strong facial detail and natural lighting.
- Download full Diffusers pipeline:
  ```bash
  mkdir -p "$HOME/sd-models/Stable-diffusion/AbsoluteReality_v1.8.1"
  hf download digiplay/AbsoluteReality_v1.8.1 --repo-type model --include "*" \
    --local-dir "$HOME/sd-models/Stable-diffusion/AbsoluteReality_v1.8.1"
  ```

## CopaxRealistic Vision v6 (community mirror, SD 1.5)
- Known for sharp portraits and flexible styling.
- Download checkpoint:
  ```bash
  hf download Googh23/Realistic_Vision_v6 Realistic_Vision_V6.safetensors \
    --repo-type model --local-dir "$HOME/sd-models/Stable-diffusion"
  ```

## RevAnimated v1.2.2 (SD 1.5)
- Hybrid realism/illustrative model for expressive characters.
- Download checkpoint:
  ```bash
  wget --header="Authorization: Bearer $HF_TOKEN" \
    "https://huggingface.co/andite/anything-v4.0/resolve/main/RevAnimated_v1.2.2.safetensors" \
    -O "$HOME/sd-models/Stable-diffusion/RevAnimated_v1.2.2.safetensors"
  ```

## Usage Tips
- Prefer `aria2c -x16 -s16` for faster large downloads when available.
- After downloading, update your model index or pipeline config to point at the new files.
- Combine these checkpoints with portrait-focused LoRAs for enhanced detail.
