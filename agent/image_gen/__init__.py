"""
Image generation package — modular split from the original image_generation.py.

Submodules:
    core        — shared globals, progress tracking, VRAM, cancellation
    models      — pipeline loading, model discovery, VRAM eviction
    lora        — LoRA management, trigger words, categories, search
    styles      — art style presets and application
    prompts     — prompt analysis, encoding, validation, vocabulary, multi-character structuring
    generation  — core image generation, regional conditioning, preview

The remaining functions (animation, video, training, upscale, storyboard)
still live in agent/image_generation.py and will be migrated incrementally.

Usage:
    # New-style imports (preferred for new code):
    from agent.image_gen.lora import get_trigger_words, parse_outfit_groups
    from agent.image_gen.models import list_models, flush_vram
    from agent.image_gen.styles import get_art_styles, ART_STYLES

    # Legacy imports still work via agent/image_generation.py
"""

# Re-export the modular public API
from .core import (
    MODELS_DIR, SUPPORTED_EXTENSIONS, LORA_EXTENSIONS,
    get_generation_progress, get_gpu_info, cancel_generation,
    _save_json, _update_progress, _check_vram_safe, _check_cancelled,
)
from .models import (
    list_models, get_active_model, _load_pipeline,
    evict_ollama_models, flush_vram, unload_model, delete_model,
    _resolve_model_path, _is_lora_file,
)
from .lora import (
    _load_trigger_words, _save_trigger_words,
    parse_trigger_word_entry, format_trigger_word,
    parse_outfit_groups, build_trigger_prompt,
    get_trigger_words, get_trigger_words_parsed,
    set_trigger_words, delete_trigger_words,
    get_all_trigger_words, list_loras_by_category,
    load_lora, unload_loras,
)
from .styles import (
    ART_STYLES, STYLE_NAMES,
    get_art_styles, _apply_art_style,
)
from .prompts import (
    _fuzzy_match_score, validate_prompt_characters,
    count_tokens, _chunk_prompt, _encode_long_prompt, _manual_chunk_encode,
    _split_respecting_parens, _analyze_prompt,
    expand_vocabulary, _VOCAB_EXPANSION,
    _structure_multi_angle_prompt, _structure_multi_character_prompt,
    _encode_single_sdxl, _create_spatial_masks,
    _parse_regional_sections, _build_anti_bleed_negative,
)
from .generation import (
    DEFAULT_NEGATIVE, EXPLICIT_NEGATIVE,
    generate_image, _generate_regional,
    _load_img2img_pipeline,
    generate_preview_only, generate_with_preview,
)
