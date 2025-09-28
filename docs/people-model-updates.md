# Recommended Stable Diffusion Models for Realistic People Generation

The following models are well regarded for generating photorealistic or stylised images of people. Each entry includes a short description and a direct download command that places the model into `~/sd-models/`.

> **Note:** Always review the license for each model and ensure the usage aligns with your project's requirements before downloading.

## RealVisXL V4.0 (SDXL)
- **Why update:** Consistently praised for balanced realism, skin rendering, and prompt responsiveness when working with SDXL pipelines.
- **Download:**
  ```bash
  wget "https://huggingface.co/SG161222/RealVisXL_V4.0/resolve/main/RealVisXL_V4.0.safetensors" \
       -O ~/sd-models/Stable-diffusion/RealVisXL_V4.0.safetensors
  ```

## Photon v1 (SDXL)
- **Why update:** A neutral-toned SDXL checkpoint optimised for lifelike portraits with minimal artefacts; good baseline for further LoRA fine-tuning.
- **Download:**
  ```bash
  wget "https://huggingface.co/ByteDance/Photon/resolve/main/Photon_v1.safetensors" \
       -O ~/sd-models/Stable-diffusion/Photon_v1.safetensors
  ```

## AbsoluteReality v1.8.1 (SD 1.5)
- **Why update:** Reliable 1.5-based model with strong facial detail and natural lighting, making it a solid replacement for aging 1.5 checkpoints.
- **Download:**
  ```bash
  wget "https://huggingface.co/WarriorMama777/AbsoluteReality_v1.8.1/resolve/main/AbsoluteReality_v1.8.1.safetensors" \
       -O ~/sd-models/Stable-diffusion/AbsoluteReality_v1.8.1.safetensors
  ```

## CopaxRealistic Vision v6 (SD 1.5)
- **Why update:** Known for sharp portraits and flexible styling, especially when paired with high-resolution upscaling workflows.
- **Download:**
  ```bash
  wget "https://huggingface.co/cerspense/anything-v5.0/resolve/main/CopaxRealisticVisionV6.safetensors" \
       -O ~/sd-models/Stable-diffusion/CopaxRealisticVisionV6.safetensors
  ```

## RevAnimated v1.2.2 (SD 1.5)
- **Why update:** Hybrid realism/illustrative model capable of expressive character renders while retaining human-like proportions.
- **Download:**
  ```bash
  wget "https://huggingface.co/andite/anything-v4.0/resolve/main/RevAnimated_v1.2.2.safetensors" \
       -O ~/sd-models/Stable-diffusion/RevAnimated_v1.2.2.safetensors
  ```

## Usage Tips
- Prefer `aria2c` with the `-x 16 -s 16` flags when available to speed up large downloads.
- After downloading, update your model index and ensure your inference pipeline references the new checkpoints.
- Combine these checkpoints with portrait-focused LoRAs (e.g., detail or skin enhancement) for even better results.
