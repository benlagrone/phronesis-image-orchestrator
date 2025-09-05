from fastapi import FastAPI, Request, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import base64
from io import BytesIO
import json
from urllib.parse import parse_qs
import os
import uuid
import time
import logging
from starlette.staticfiles import StaticFiles

log_level = os.getenv("LOG_LEVEL", "info").upper()
level_value = getattr(logging, log_level, logging.INFO)

# Avoid double logging: let uvicorn configure root; configure only our app logger
app_logger = logging.getLogger("sd-api")
app_logger.setLevel(level_value)
if not app_logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
    app_logger.addHandler(_h)
# Don't propagate to root to prevent duplicate lines
app_logger.propagate = False

app = FastAPI()
app.mount("/files", StaticFiles(directory="/output"), name="files")
logger = app_logger
_pipeline_cache: Dict[str, object] = {}
VERBOSE_REQUEST_LOG = os.getenv("VERBOSE_REQUEST_LOG", "0").lower() in ("1", "true", "yes", "on")


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
    vae: Optional[str] = None    # optional VAE filename in /models/VAE or HF repo id

    # Accepted but currently unused (placeholders for compatibility)
    sampler_index: Optional[str] = None
    schedule_type: Optional[str] = None
    enable_hr: Optional[bool] = None
    hr_scale: Optional[float] = None
    hr_upscaler: Optional[str] = None
    denoising_strength: Optional[float] = None
    refiner_switch_at: Optional[float] = None
    refiner_checkpoint: Optional[str] = None
    model_hash: Optional[str] = None
    version: Optional[str] = None

def _resolve_local_file(name: str, search_dirs: list[str]) -> Optional[str]:
    if not name:
        return None
    if os.path.isabs(name) and os.path.exists(name):
        return name
    for d in search_dirs:
        candidate = os.path.join(d, name)
        if os.path.exists(candidate):
            return candidate
    return None


def _load_pipeline(model_ref: Optional[str] = None, vae_ref: Optional[str] = None):
    from diffusers import AutoPipelineForText2Image, AutoencoderKL, StableDiffusionPipeline, StableDiffusionXLPipeline
    import torch

    # Determine reference: explicit payload model, else env SD_MODEL
    ref = model_ref or os.getenv("SD_MODEL", "runwayml/stable-diffusion-v1-5")
    use_cuda = torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32

    # If looks like a local single file (.ckpt/.safetensors), resolve against known model dirs
    if isinstance(ref, str) and (ref.endswith(".ckpt") or ref.endswith(".safetensors")):
        candidate = _resolve_local_file(ref, [
            "/models/Stable-diffusion",
            "/models",
        ])
        if not candidate:
            raise HTTPException(status_code=400, detail=f"Model file not found: /models/Stable-diffusion/{ref}")
        key = f"single:{candidate}"
        if key in _pipeline_cache:
            pipe = _pipeline_cache[key]
        else:
            logger.info(f"Loading pipeline (auto) from single file: {candidate}")
            # Auto-detect SD v1 vs SDXL
            try:
                pipe = AutoPipelineForText2Image.from_single_file(candidate, torch_dtype=dtype)
            except Exception:
                # Fallback to explicit pipelines
                try:
                    pipe = StableDiffusionXLPipeline.from_single_file(candidate, torch_dtype=dtype)
                except Exception:
                    pipe = StableDiffusionPipeline.from_single_file(candidate, torch_dtype=dtype)
            _pipeline_cache[key] = pipe
    else:
        # HF repo id or local directory
        key = f"pretrained:{ref}"
        if key in _pipeline_cache:
            pipe = _pipeline_cache[key]
        else:
            logger.info(f"Loading pipeline (auto) from pretrained: {ref}")
            pipe = AutoPipelineForText2Image.from_pretrained(ref, torch_dtype=dtype)
            _pipeline_cache[key] = pipe

    if use_cuda:
        pipe.to("cuda")
        logger.info("Moved pipeline to CUDA")
    else:
        logger.info("Using CPU for inference (float32)")

    # Optional VAE override
    if vae_ref:
        vae_path = None
        if isinstance(vae_ref, str) and (vae_ref.endswith(".safetensors") or vae_ref.endswith(".ckpt") or os.path.isdir(vae_ref)):
            vae_path = _resolve_local_file(vae_ref, [
                "/models/VAE",
                "/models/vae",
                "/models/Stable-diffusion",
                "/models",
            ]) or (vae_ref if os.path.isabs(vae_ref) and os.path.exists(vae_ref) else None)
        try:
            if vae_path and os.path.exists(vae_path):
                logger.info(f"Loading VAE from file/dir: {vae_path}")
                vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=dtype)
                pipe.vae = vae
            elif isinstance(vae_ref, str) and not vae_path:
                logger.info(f"Loading VAE from pretrained repo: {vae_ref}")
                vae = AutoencoderKL.from_pretrained(vae_ref, torch_dtype=dtype)
                pipe.vae = vae
        except Exception as e:
            logger.warning(f"Failed to load VAE '{vae_ref}': {e}")

    return pipe

@app.get("/health")
def health():
    return {"ok": True}


@app.get("/sdapi/v1/models")
def list_models():
    base_dir = "/models/Stable-diffusion"
    exts = (".safetensors", ".ckpt")
    items = []
    try:
        for name in sorted(os.listdir(base_dir)):
            if not name.lower().endswith(exts):
                continue
            path = os.path.join(base_dir, name)
            try:
                size = os.path.getsize(path)
            except Exception:
                size = None
            items.append({"model_name": name, "path": path, "size_bytes": size})
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail=f"Models directory not found: {base_dir}")
    return {"ok": True, "count": len(items), "models": items}

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
    try:
        p = _load_pipeline(payload.model, payload.vae)
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
    except HTTPException:
        # already structured; just bubble up
        raise
    except Exception as e:
        logger.exception("txt2img failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.exception_handler(HTTPException)
async def http_exception_logger(request: Request, exc: HTTPException):
    try:
        body = await request.body()
        body_preview = body[:500].decode("utf-8", errors="ignore") if body else ""
    except Exception:
        body_preview = "<unavailable>"
    logger.warning(
        "HTTP %s at %s: %s | body: %s",
        exc.status_code,
        request.url.path,
        str(exc.detail),
        body_preview,
    )
    return JSONResponse(status_code=exc.status_code, content={"ok": False, "detail": exc.detail})


@app.exception_handler(RequestValidationError)
async def validation_exception_logger(request: Request, exc: RequestValidationError):
    try:
        body = await request.body()
        body_preview = body[:500].decode("utf-8", errors="ignore") if body else ""
    except Exception:
        body_preview = "<unavailable>"
    logger.warning(
        "Validation error at %s: %s | body: %s",
        request.url.path,
        exc.errors(),
        body_preview,
    )
    def _sanitize(obj: Any) -> Any:
        if isinstance(obj, bytes):
            return obj.decode("utf-8", errors="ignore")
        if isinstance(obj, list):
            return [_sanitize(x) for x in obj]
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        return obj

    return JSONResponse(
        status_code=422,
        content={"ok": False, "detail": "Validation error", "errors": _sanitize(exc.errors())},
    )


@app.middleware("http")
async def request_timer(request: Request, call_next):
    t0 = time.time()
    resp = await call_next(request)
    dt = time.time() - t0
    logger.info("REQ %s %s -> %s in %.2fs", request.method, request.url.path, getattr(resp, "status_code", "-"), dt)
    return resp


async def _coerce_payload(request: Request) -> Txt2ImgPayload:
    """Accept JSON, form-encoded, or query-only payloads and return a Txt2ImgPayload."""
    data: Dict[str, Any] = {}
    ct = (request.headers.get("content-type") or "").lower()

    # Read raw body once
    raw: bytes = b""
    try:
        raw = await request.body()
    except Exception:
        raw = b""

    # 1) Try JSON from raw
    if raw:
        try:
            j = json.loads(raw)
            if isinstance(j, dict):
                data = j
        except Exception:
            pass

    # 2) Try urlencoded parsing if still empty
    if not data and raw:
        try:
            parsed = parse_qs(raw.decode("utf-8", errors="ignore"), keep_blank_values=True)
            flat: Dict[str, Any] = {k: v[-1] if isinstance(v, list) else v for k, v in parsed.items()}
            if flat:
                data = flat
        except Exception:
            pass

    # 3) As a last resort, try Starlette's form parser (multipart, etc.)
    if not data and ("multipart/form-data" in ct or "application/x-www-form-urlencoded" in ct or not ct):
        try:
            form = await request.form()
            data = {k: v for k, v in form.items()}
        except Exception:
            pass

    # 4) Merge query params without overwriting body
    try:
        for k, v in request.query_params.items():
            data.setdefault(k, v)
    except Exception:
        pass

    # Unwrap nested payloads like { "image": { ... } } or { "data": { ... } }
    if isinstance(data, dict):
        if "image" in data and isinstance(data["image"], dict):
            data = data["image"]
        elif "data" in data and isinstance(data["data"], dict):
            data = data["data"]
    # If form had a single field named 'image' containing JSON, try to parse it
    if isinstance(data, dict) and not data:
        for wrapper in ("image", "data"):
            if wrapper in request.query_params:
                try:
                    maybe = json.loads(request.query_params[wrapper])  # type: ignore[arg-type]
                    if isinstance(maybe, dict):
                        data = maybe
                        break
                except Exception:
                    pass
    # Also check if earlier form parsing yielded a single 'image' key with JSON string
    if isinstance(data, dict) and len(data) == 1:
        for wrapper in ("image", "data"):
            if wrapper in data and isinstance(data[wrapper], str):
                try:
                    maybe = json.loads(data[wrapper])  # type: ignore[arg-type]
                    if isinstance(maybe, dict):
                        data = maybe
                        break
                except Exception:
                    pass

    # Map legacy names
    if "guidance" in data and "cfg_scale" not in data:
        data["cfg_scale"] = data.get("guidance")

    # Coerce numerics
    for key in ("width", "height", "steps", "batch_size", "batch_count"):
        if key in data and data[key] not in (None, ""):
            try:
                data[key] = int(float(data[key]))
            except Exception:
                pass
    if "cfg_scale" in data and data["cfg_scale"] not in (None, ""):
        try:
            data["cfg_scale"] = float(data["cfg_scale"])
        except Exception:
            pass
    if "seed" in data and data["seed"] not in (None, ""):
        try:
            data["seed"] = int(float(data["seed"]))
        except Exception:
            pass

    # Ensure required field presence with a clearer error if missing
    if not data.get("prompt"):
        raise HTTPException(status_code=422, detail="Missing required field: prompt (send as JSON or form)")

    try:
        if VERBOSE_REQUEST_LOG:
            preview = str(data.get("prompt", ""))[:140].replace("\n", " ")
            logger.debug(
                "Parsed payload keys=%s size=%sx%s steps=%s cfg=%s seed=%s model=%s | prompt='%'",
                sorted(list(data.keys())),
                data.get("width"), data.get("height"), data.get("steps"), data.get("cfg_scale"), data.get("seed"), data.get("model"),
                preview,
            )
        return Txt2ImgPayload(**data)
    except Exception:
        # Convert Pydantic errors to HTTP 422 with clearer detail
        keys = ",".join(sorted(map(str, data.keys())))
        raise HTTPException(status_code=422, detail=f"Invalid request body. Parsed keys: [{keys}]")


@app.post("/txt2img")
async def txt2img_legacy(request: Request):
    # Accept JSON or form-encoded and return A1111-compatible response
    payload = await _coerce_payload(request)
    return await txt2img_automatic1111_compat(request, payload)


@app.post("/sdapi/v1/txt2img")
async def txt2img_automatic1111_compat(request: Request, payload: Optional[Txt2ImgPayload] = None):
    # A1111-compatible response shape: { images: [base64, ...], parameters, info }
    t0 = time.time()
    if payload is None:
        payload = await _coerce_payload(request)
    if VERBOSE_REQUEST_LOG:
        preview = (payload.prompt or "")[:140].replace("\n", " ")
        logger.info(
            "txt2img req size=%dx%d steps=%d cfg=%.2f seed=%s model=%s batch=%dx%d | prompt='%'",
            payload.width, payload.height, payload.steps, payload.cfg_scale, str(payload.seed), str(payload.model), payload.batch_count, payload.batch_size, preview,
        )
    p = _load_pipeline(payload.model, payload.vae)
    items = _generate_and_save(p, payload)
    # Encode saved PNGs to base64
    images_b64: List[str] = []
    for it in items:
        try:
            with open(it["path"], "rb") as f:
                b = f.read()
            images_b64.append(base64.b64encode(b).decode("utf-8"))
        except Exception as e:
            logger.warning(f"Failed to base64-encode {it['path']}: {e}")
    duration = time.time() - t0
    logger.info(f"Generated {len(items)} image(s) in {duration:.2f}s (A1111 response)")
    return {
        "images": images_b64,
        "parameters": payload.dict(),
        "info": "",
    }
