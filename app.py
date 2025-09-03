from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Optional, List, Dict
import os
import uuid
import time
import logging
from starlette.staticfiles import StaticFiles

app = FastAPI()
app.mount("/files", StaticFiles(directory="/output"), name="files")
logger = logging.getLogger("sd-api")
_pipeline_cache: Dict[str, object] = {}


class Txt2ImgPayload(BaseModel):
    # Common controls
    prompt: str
    negative_prompt: Optional[str] = None
    width: int = 512
    height: int = 512
    steps: int = 25
    cfg_scale: float = 7.5  # maps to guidance_scale
    seed: Optional[int] = None
    batch_size: int = 1
    batch_count: int = 1

    # Model selection
    model: Optional[str] = None  # filename in /models/Stable-diffusion or HF repo id

    # Accepted but currently unused (placeholders for compatibility)
    sampler_index: Optional[str] = None
    schedule_type: Optional[str] = None
    enable_hr: Optional[bool] = None
    hr_scale: Optional[float] = None
    hr_upscaler: Optional[str] = None
    denoising_strength: Optional[float] = None
    refiner_switch_at: Optional[float] = None
    model_hash: Optional[str] = None
    version: Optional[str] = None

def _load_pipeline(model_ref: Optional[str] = None):
    from diffusers import StableDiffusionPipeline
    import torch

    # Determine reference: explicit payload model, else env SD_MODEL
    ref = model_ref or os.getenv("SD_MODEL", "runwayml/stable-diffusion-v1-5")

    # If looks like a local single file (.ckpt/.safetensors), build absolute path
    if isinstance(ref, str) and (ref.endswith(".ckpt") or ref.endswith(".safetensors")):
        candidate = ref
        if not os.path.isabs(candidate):
            candidate = os.path.join("/models/Stable-diffusion", ref)
        key = f"single:{candidate}"
        if key in _pipeline_cache:
            return _pipeline_cache[key]
        logger.info(f"Loading pipeline from single file: {candidate}")
        pipe = StableDiffusionPipeline.from_single_file(candidate, torch_dtype=torch.float16)
    else:
        # HF repo id or local directory
        key = f"pretrained:{ref}"
        if key in _pipeline_cache:
            return _pipeline_cache[key]
        logger.info(f"Loading pipeline from pretrained: {ref}")
        pipe = StableDiffusionPipeline.from_pretrained(ref, torch_dtype=torch.float16)

    if torch.cuda.is_available():
        pipe.to("cuda")
        logger.info("Moved pipeline to CUDA")
    else:
        logger.info("Using CPU for inference")

    _pipeline_cache[key] = pipe
    return pipe

@app.get("/health")
def health():
    return {"ok": True}

def _generate_and_save(pipeline, payload: Txt2ImgPayload) -> List[Dict[str, str]]:
    import torch

    # Setup seed/generator
    generator = None
    if payload.seed is not None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=device).manual_seed(int(payload.seed))

    os.makedirs("/output", exist_ok=True)
    results: List[Dict[str, str]] = []

    for _ in range(max(1, int(payload.batch_count))):
        out = pipeline(
            prompt=payload.prompt,
            negative_prompt=payload.negative_prompt,
            width=int(payload.width),
            height=int(payload.height),
            num_inference_steps=int(payload.steps),
            guidance_scale=float(payload.cfg_scale),
            generator=generator,
            num_images_per_prompt=max(1, int(payload.batch_size)),
        )
        for img in out.images:
            filename = f"out-{uuid.uuid4().hex}.png"
            out_path = os.path.join("/output", filename)
            img.save(out_path)
            results.append({"path": out_path, "filename": filename})
    return results


def _txt2img_impl(request: Request, payload: Txt2ImgPayload):
    t0 = time.time()
    p = _load_pipeline(payload.model)
    logger.info(
        "txt2img json size=%dx%d steps=%d cfg=%.2f seed=%s model=%s",
        payload.width,
        payload.height,
        payload.steps,
        payload.cfg_scale,
        str(payload.seed),
        str(payload.model),
    )

    items = _generate_and_save(p, payload)
    duration = time.time() - t0
    base = str(request.base_url).rstrip("/")
    urls = [f"{base}/files/{it['filename']}" for it in items]
    logger.info(f"Generated {len(items)} image(s) in {duration:.2f}s")
    return {"ok": True, "count": len(items), "paths": [it["path"] for it in items], "urls": urls}


@app.post("/txt2img")
def txt2img_legacy(request: Request, payload: Txt2ImgPayload):
    # Maintain old path but switch to JSON body
    return _txt2img_impl(request, payload)


@app.post("/sdapi/v1/txt2img")
def txt2img_automatic1111_compat(request: Request, payload: Txt2ImgPayload):
    # New JSON endpoint similar to A1111 path
    return _txt2img_impl(request, payload)
