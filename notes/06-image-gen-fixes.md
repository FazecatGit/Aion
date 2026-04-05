# Image Generation Fixes

---

## Issues Fixed

### 1. Character LoRAs Not Generating Correctly
- **Problem**: Only first trigger word was injected
- **Fix**: ALL trigger words are now injected via `build_trigger_prompt()`

### 2. Preview Not Noisy
- **Problem**: Preview was a completely different image, not a noisy version of final
- **Fix**: Rewrote `generate_preview_only()` to use `callback_on_step_end` capturing intermediate noisy latent at step 3-4. Same seed guarantees same composition as full render

### 3. Feedback Not Applied
- **Problem**: Positive feedback system wasn't being used
- **Fix**: Added positive feedback system alongside existing negatives. `apply_positive_learnings()` called in all 3 generation paths (image, animated, preview)

### 4. Trigger Word Weights
- New `parse_trigger_word_entry()` and `format_trigger_word()` support `{"word": str, "weight": float}` format

### 5. Semicolon Outfit Groups
- `parse_outfit_groups()` splits trigger words by `;` separator into primary + outfit groups

---

## Files Modified
- `agent/image_generation.py`: New functions (parse_trigger_word_entry, format_trigger_word, parse_outfit_groups, build_trigger_prompt, get_trigger_words_parsed), positive feedback system
- `api.py`: Updated request models, new endpoints (feedback/learnings, feedback/clear, feedback/positive, loras/trigger-words/parsed)
- `frontend/orb-app/src/renderer.tsx`: LoRA weight sliders, outfit selector, LoRA diagnostics display, feedback learnings panel

---

## Key Design Decisions
- `generate_image()` has `selected_outfits: dict[str, str] | None` param
- LoRA diagnostics returned in response as `lora_diagnostics` list and `injected_triggers` string
- Frontend persists `loraWeights` and `selectedOutfits` to localStorage

## Key Learnings
- Always inject ALL trigger words, not just the first — multiple triggers define the character
- Preview should use same seed + early interruption, not a separate low-res generation
- Outfit groups separated by `;` allow flexible character customization
