from fastapi import FastAPI
from pydantic import BaseModel
from diffusers import StableDiffusionPipeline
import torch
import os
from pathlib import Path

app = FastAPI()

class Txt2ImgRequest(BaseModel):
    prompt: str
    negative_prompt: str | None = None
    width: int = 512
    height: int = 512
    steps: int = 20
    cfg_scale: float = 7.5

MODEL_ID = os.environ.get("MODEL_ID", "runwayml/stable-diffusion-v1-5")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
)
if DEVICE == "cuda":
    pipe = pipe.to("cuda")

@app.post("/sdapi/v1/txt2img")
def txt2img(request: Txt2ImgRequest):
    images = pipe(
        request.prompt,
        negative_prompt=request.negative_prompt,
        width=request.width,
        height=request.height,
        num_inference_steps=request.steps,
        guidance_scale=request.cfg_scale,
    ).images
    output_dir = Path("/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "output.png"
    images[0].save(output_path)
    return {"filename": str(output_path)}
