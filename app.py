import asyncio

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

import torch
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

_generation_lock = asyncio.Semaphore(max(1, int(os.getenv("SD_MAX_CONCURRENT", "1"))))
SD_SDXL_MAX_PIXELS = int(os.getenv("SD_SDXL_MAX_PIXELS", "600000"))
SD_DEFAULT_MAX_PIXELS = int(os.getenv("SD_MAX_PIXELS", "1200000"))


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


def _is_sdxl_reference(model_ref: Optional[str]) -> bool:
    if not model_ref:
        return False
    name = model_ref.lower()
    return "sdxl" in name or "stable-diffusion-xl" in name or name.endswith("-xl")


def _effective_model_ref(payload_model: Optional[str]) -> str:
    return payload_model or os.getenv("SD_MODEL", "runwayml/stable-diffusion-v1-5")


def _guard_payload_limits(payload: Txt2ImgPayload) -> None:
    model_ref = _effective_model_ref(payload.model)
    width = int(payload.width)
    height = int(payload.height)
    batch = max(1, int(payload.batch_size) * int(payload.batch_count))
    total_pixels = width * height * batch

    limit = SD_SDXL_MAX_PIXELS if _is_sdxl_reference(model_ref) else SD_DEFAULT_MAX_PIXELS

    if total_pixels > limit:
        max_side = 1024 if _is_sdxl_reference(model_ref) else 832
        factor_w = min(1.0, max_side / width)
        factor_h = min(1.0, max_side / height)
        factor = min(factor_w, factor_h, (limit / max(width * height, 1)) ** 0.5)
        payload.width = max(64, int(width * factor) // 8 * 8)
        payload.height = max(64, int(height * factor) // 8 * 8)
        payload.batch_size = 1
        payload.batch_count = 1


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

    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    if hasattr(pipe, "enable_vae_slicing"):
        try:
            pipe.enable_vae_slicing()
        except Exception:
            logger.debug("enable_vae_slicing not available on pipeline")
    if os.getenv("SD_ENABLE_XFORMERS", "1").lower() not in ("0", "false", "no", "off") and hasattr(pipe, "enable_xformers_memory_efficient_attention"):
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception as exc:
            logger.info(f"xFormers attention not enabled: {exc}")

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

    # Unwrap payloads that are entirely nested under a single wrapper key
    if isinstance(data, dict):
        for wrapper_key in ("image", "data", "payload"):
            if set(data.keys()) == {wrapper_key} and isinstance(data.get(wrapper_key), dict):
                data = data[wrapper_key]
                break
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

    # Normalize structured request bodies (prompt/model/settings grouping)
    if isinstance(data, dict):
        def pick(d: Optional[Dict[str, Any]], *keys: str) -> Optional[Any]:
            if not isinstance(d, dict):
                return None
            for key in keys:
                if key in d and d[key] not in (None, ""):
                    return d[key]
            return None

        prompt_block = data.get("prompt") if isinstance(data.get("prompt"), dict) else None
        if prompt_block:
            main_prompt = pick(prompt_block, "text", "positive", "value", "prompt")
            if isinstance(main_prompt, str) and main_prompt.strip():
                data["prompt"] = main_prompt
            else:
                fallback_parts = [v.strip() for v in prompt_block.values() if isinstance(v, str) and v.strip()]
                if fallback_parts:
                    data["prompt"] = "\n".join(fallback_parts)
            neg_prompt = pick(prompt_block, "negative", "negative_prompt", "negativePrompt", "negativeText")
            if isinstance(neg_prompt, str) and neg_prompt.strip() and not isinstance(data.get("negative_prompt"), str):
                data["negative_prompt"] = neg_prompt

        settings_block: Optional[Dict[str, Any]] = None
        for key in ("settings", "params", "config", "generation", "options"):
            candidate = data.get(key)
            if isinstance(candidate, dict):
                settings_block = candidate
                break

        size_sources: list[Dict[str, Any]] = []
        for source in (data.get("image"), data.get("render"), data.get("size"), settings_block):
            if isinstance(source, dict):
                size_sources.append(source)
                for nested_key in ("size", "dimensions", "resolution", "shape"):
                    nested = source.get(nested_key)
                    if isinstance(nested, dict):
                        size_sources.append(nested)

        def pull_numeric(field: str, aliases: tuple[str, ...]) -> None:
            if field in data and data[field] not in (None, ""):
                return
            for src in size_sources:
                val = pick(src, *aliases)
                if val not in (None, ""):
                    data[field] = val
                    return

        pull_numeric("width", ("width", "w"))
        pull_numeric("height", ("height", "h"))

        sampling_sources: list[Dict[str, Any]] = []
        for source in (data.get("sampling"), data.get("sampler"), settings_block):
            if isinstance(source, dict):
                sampling_sources.append(source)
                for nested_key in ("sampling", "sampler", "steping"):
                    nested = source.get(nested_key)
                    if isinstance(nested, dict):
                        sampling_sources.append(nested)

        def pull_from_sources(field: str, aliases: tuple[str, ...], sources: list[Dict[str, Any]]) -> None:
            if field in data and data[field] not in (None, ""):
                return
            for src in sources:
                val = pick(src, *aliases)
                if val not in (None, ""):
                    data[field] = val
                    return

        pull_from_sources("steps", ("steps", "step_count"), sampling_sources)
        pull_from_sources("cfg_scale", ("cfg", "cfg_scale", "guidance", "guidance_scale"), sampling_sources)
        pull_from_sources("seed", ("seed", "random_seed"), sampling_sources)

        batch_sources: list[Dict[str, Any]] = []
        for source in (data.get("batch"), settings_block):
            if isinstance(source, dict):
                batch_sources.append(source)
                nested = source.get("batch") if isinstance(source.get("batch"), dict) else None
                if nested:
                    batch_sources.append(nested)

        pull_from_sources("batch_size", ("size", "batch_size", "per_batch"), batch_sources)
        pull_from_sources("batch_count", ("count", "batch_count", "batches"), batch_sources)

        model_block = data.get("model") if isinstance(data.get("model"), dict) else None
        if model_block:
            model_id = pick(model_block, "id", "name", "path", "model", "identifier")
            data["model"] = model_id if model_id not in (None, "") else None
            vae_id = pick(model_block, "vae", "vae_id", "vae_name")
            if vae_id not in (None, "") and (data.get("vae") in (None, "") or isinstance(data.get("vae"), dict)):
                data["vae"] = vae_id
        elif isinstance(data.get("model"), dict):
            data["model"] = None

        vae_block = data.get("vae") if isinstance(data.get("vae"), dict) else None
        if vae_block:
            vae_id = pick(vae_block, "id", "name", "path", "identifier", "vae")
            data["vae"] = vae_id if vae_id not in (None, "") else None

    if isinstance(data.get("prompt"), dict):
        raise HTTPException(status_code=422, detail="Invalid prompt block; expected string text field")

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
    async with _generation_lock:
        t0 = time.time()
        if payload is None:
            payload = await _coerce_payload(request)
        _guard_payload_limits(payload)
        if VERBOSE_REQUEST_LOG:
            preview = (payload.prompt or "")[:140].replace("\n", " ")
            logger.info(
                "txt2img req size=%dx%d steps=%d cfg=%.2f seed=%s model=%s batch=%dx%d | prompt='%'",
                payload.width, payload.height, payload.steps, payload.cfg_scale, str(payload.seed), str(payload.model), payload.batch_count, payload.batch_size, preview,
            )
        try:
            p = _load_pipeline(payload.model, payload.vae)
            items = _generate_and_save(p, payload)
        except RuntimeError as exc:
            message = str(exc)
            if "CUDA out of memory" in message:
                logger.warning("Generation rejected due to CUDA OOM: %s", message)
                if torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        logger.debug("Failed to empty CUDA cache after OOM", exc_info=True)
                raise HTTPException(
                    status_code=503,
                    detail="GPU out of memory for requested parameters. Reduce size or batch and retry.",
                ) from exc
            raise
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
