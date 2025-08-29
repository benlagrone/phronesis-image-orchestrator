from fastapi import FastAPI, Request
from typing import Optional
import os
import uuid
import time
import logging
from starlette.staticfiles import StaticFiles

app = FastAPI()
app.mount("/files", StaticFiles(directory="/output"), name="files")
logger = logging.getLogger("sd-api")
pipe = None

def get_pipe():
    global pipe
    if pipe is None:
        from diffusers import StableDiffusionPipeline
        import torch
        model_id = os.getenv("SD_MODEL", "runwayml/stable-diffusion-v1-5")
        logger.info(f"Loading model: {model_id}")
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        if torch.cuda.is_available():
            pipe.to("cuda")
            logger.info("Moved pipeline to CUDA")
        else:
            logger.info("Using CPU for inference")
    return pipe

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/txt2img")
def txt2img(request: Request,
            prompt: str,
            negative_prompt: Optional[str] = None,
            width: int = 512,
            height: int = 512,
            steps: int = 25,
            guidance: float = 7.5):
    p = get_pipe()
    t0 = time.time()
    logger.info(f"txt2img prompt='{prompt}' neg='{negative_prompt}' size={width}x{height} steps={steps} guidance={guidance}")
    image = p(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=guidance
    ).images[0]
    os.makedirs("/output", exist_ok=True)
    filename = f"out-{uuid.uuid4().hex}.png"
    out_path = os.path.join("/output", filename)
    image.save(out_path)
    duration = time.time() - t0
    # Build absolute URL for convenience
    base = str(request.base_url).rstrip("/")
    url = f"{base}/files/{filename}"
    logger.info(f"Generated image -> {out_path} in {duration:.2f}s; url={url}")
    return {"ok": True, "path": out_path, "url": url}
