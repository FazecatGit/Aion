# Regional Conditioning Implementation

---

## Overview
Regional conditioning allows multi-character images with spatial isolation — each character gets their own region of the image without bleeding into others.

---

## New Functions (image_generation.py)

### `_encode_single_sdxl(pipe, prompt, device)`
- Encodes one prompt with SDXL's dual text encoders (CLIP-L + CLIP-G)
- Returns (prompt_embeds, pooled)

### `_create_spatial_masks(regions, latent_w, latent_h, device, dtype)`
- Creates soft Gaussian-edged masks for left/right/center/background positions

### `_parse_regional_sections(prompt)`
- Parses BREAK-separated prompt into (shared_prompt, regions list)
- Returns None if not multi-character

### `_generate_regional(pipe, prompt, negative_prompt, width, height, steps, guidance_scale, generator)`
- Custom denoising loop: per-step runs UNet for each character region separately
- Blends noise predictions with spatial masks
- N+2 UNet passes per step (~2.5x slower for 3 chars)

### `_build_anti_bleed_negative(regions)`
- Scans character descriptions for color features (hair/eye colors)
- Generates cross-character negative prompt additions

---

## How It Works
1. Detects multi-character BREAK prompts (shared + 2+ character sections)
2. Encodes each section separately via dual SDXL encoders
3. Creates soft spatial masks based on position hints ("on the left", "on the right")
4. Per denoising step: unconditional pass + shared pass (background) + N character passes, blended by masks
5. CFG applied to blended noise prediction
6. Anti-bleed negative prompt auto-generated from color features

---

## Hook Point
In `generate_image()`, before `pipe(**gen_kwargs)`, checks `_parse_regional_sections(full_prompt)`. If multi-character, uses `_generate_regional()` instead.

## Key Learnings
- Gaussian-edged masks eliminate hard boundaries between character regions
- Anti-bleed negatives prevent hair/eye color from leaking between characters
- ~2.5x slowdown is acceptable for proper multi-character isolation
- Shared prompt section (before first BREAK) applies to entire image as background context
