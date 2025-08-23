from fastapi import FastAPI
from typing import Optional
import os

app = FastAPI()
pipe = None

def get_pipe():
    global pipe
    if pipe is None:
        from diffusers import StableDiffusionPipeline
        import torch
        model_id = os.getenv("SD_MODEL", "runwayml/stable-diffusion-v1-5")
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        if torch.cuda.is_available():
            pipe.to("cuda")
    return pipe

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/txt2img")
def txt2img(prompt: str,
            negative_prompt: Optional[str] = None,
            width: int = 512,
            height: int = 512,
            steps: int = 25,
            guidance: float = 7.5):
    p = get_pipe()
    image = p(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=guidance
    ).images[0]
    out = "/tmp/out.png"
    image.save(out)
    return {"ok": True, "path": out}
