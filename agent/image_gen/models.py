"""Pipeline loading, model discovery, VRAM eviction."""

import gc
import logging

import torch
from diffusers import StableDiffusionXLPipeline

from .core import (
    MODELS_DIR, SUPPORTED_EXTENSIONS, LORA_EXTENSIONS,
    _loaded_pipelines, _active_model, _active_loras, _log,
)
from .lora import get_trigger_words


def list_models() -> list[dict]:
    """Discover all checkpoint models and LoRAs in the models folder."""
    models = []
    if not MODELS_DIR.exists():
        return models

    for f in sorted(MODELS_DIR.iterdir()):
        if f.suffix.lower() in SUPPORTED_EXTENSIONS and f.is_file():
            is_lora = _is_lora_file(f)
            models.append({
                "name": f.stem,
                "filename": f.name,
                "path": str(f),
                "size_mb": round(f.stat().st_size / (1024 * 1024), 1),
                "type": "lora" if is_lora else "checkpoint",
            })

    LORA_CATEGORIES = ("styles", "characters", "clothing", "poses", "concept", "action", "loras")
    for subdir_name in LORA_CATEGORIES:
        sub_dir = MODELS_DIR / subdir_name
        if sub_dir.exists():
            for f in sorted(sub_dir.iterdir()):
                if f.suffix.lower() in LORA_EXTENSIONS and f.is_file():
                    tw = get_trigger_words(f.stem)
                    models.append({
                        "name": f.stem,
                        "filename": f.name,
                        "path": str(f),
                        "size_mb": round(f.stat().st_size / (1024 * 1024), 1),
                        "type": "lora",
                        "category": subdir_name,
                        "trigger_words": tw,
                    })

    return models


def _is_lora_file(path) -> bool:
    """Heuristic: LoRA files are generally < 300 MB."""
    return path.stat().st_size < 300 * 1024 * 1024


def get_active_model() -> str | None:
    import agent.image_gen.core as _core
    return _core._active_model


def _load_pipeline(model_path: str) -> StableDiffusionXLPipeline:
    """Load or retrieve a cached pipeline for the given checkpoint."""
    import agent.image_gen.core as _core

    if model_path in _core._loaded_pipelines:
        _core._active_model = model_path
        return _core._loaded_pipelines[model_path]

    _log.info("[IMAGE GEN] Loading model: %s", model_path)
    pipe = StableDiffusionXLPipeline.from_single_file(
        model_path,
        torch_dtype=torch.float16,
        use_safetensors=True,
    ).to("cuda")
    pipe.enable_attention_slicing()

    _core._loaded_pipelines[model_path] = pipe
    _core._active_model = model_path
    _core._active_loras = []
    return pipe


def evict_ollama_models():
    """Evict all Ollama models from GPU so VRAM is free for image generation."""
    try:
        import httpx
        resp = httpx.get("http://localhost:11434/api/ps", timeout=5)
        if resp.status_code == 200:
            running = resp.json().get("models", [])
            for m in running:
                name = m.get("name", "")
                _log.info("[IMAGE GEN] Evicting Ollama model from GPU: %s", name)
                httpx.post(
                    "http://localhost:11434/api/generate",
                    json={"model": name, "keep_alive": 0},
                    timeout=10,
                )
    except Exception as e:
        _log.warning("[IMAGE GEN] Could not evict Ollama models: %s", e)
    gc.collect()


def flush_vram():
    """Unload ALL image-gen pipelines + evict Ollama models to fully free VRAM."""
    import agent.image_gen.core as _core

    keys = list(_core._loaded_pipelines.keys())
    for k in keys:
        del _core._loaded_pipelines[k]
    _core._active_model = None
    _core._active_loras = []

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _log.info("[IMAGE GEN] Flushed all SD pipelines from VRAM")

    evict_ollama_models()
    _log.info("[IMAGE GEN] VRAM flush complete")


def unload_model(model_path: str | None = None):
    """Free a model from GPU memory."""
    import agent.image_gen.core as _core

    target = model_path or _core._active_model
    if target and target in _core._loaded_pipelines:
        del _core._loaded_pipelines[target]
        torch.cuda.empty_cache()
        _log.info("[IMAGE GEN] Unloaded model: %s", target)
        if _core._active_model == target:
            _core._active_model = None
            _core._active_loras = []


def delete_model(model_path: str) -> dict:
    """Remove a model file from disk."""
    import agent.image_gen.core as _core
    from pathlib import Path

    p = Path(model_path)
    if not p.exists():
        return {"status": "error", "message": f"File not found: {model_path}"}
    if not str(p.resolve()).startswith(str(MODELS_DIR.resolve())):
        return {"status": "error", "message": "Cannot delete files outside models directory"}

    if model_path in _core._loaded_pipelines:
        del _core._loaded_pipelines[model_path]
        if _core._active_model == model_path:
            _core._active_model = None
            _core._active_loras = []

    p.unlink()
    _log.info("[IMAGE GEN] Deleted model: %s", model_path)
    return {"status": "ok", "deleted": model_path}


def _resolve_model_path(model_path: str | None) -> str:
    """Resolve model path, falling back to first available checkpoint."""
    if model_path:
        return model_path
    models = list_models()
    checkpoints = [m for m in models if m["type"] == "checkpoint"]
    if not checkpoints:
        raise RuntimeError("No checkpoint models found in models/ directory")
    return checkpoints[0]["path"]
