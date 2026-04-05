# Image Studio UX Overhaul — March 2026

---

## Phase 1: LoRA Trigger Word Auto-Injection
- **Backend**: Auto-injection of missing trigger words for active LoRAs into the prompt before diffusion
- **Frontend**: `toggleLoraWithTrigger()` helper auto-inserts primary trigger word when activating a LoRA and removes it when deactivating
- Dedup check prevents duplicate word insertion
- `removeTriggerWord()` for clean removal

---

## Phase 2: History Prompt Weight Preservation
- Removed aggressive regexes that stripped `(:1.1)` and `(word:1.5)` weight syntax from history prompts
- Now only strips backend-injected artifacts (BREAK, positional anchors, score tags, multi-character count tags)
- Strips known art style prefixes when restoring from history

---

## Phase 3: Multi-Character Tag Injection Fix
- Rewrote `_structure_multi_character_prompt()` to only activate with 2+ character-category LoRAs
- Removed hardcoded "multiple girls" — now uses neutral "group shot" only
- Prevents false positives from trigger words like "anime screencap" matching across all LoRA categories

---

## Phase 4: Character Preview Images
- `list_loras_by_category()` and `search_characters()` now include `preview_image` field
- Convention: `<name>.preview.png/jpg/webp` alongside `.safetensors`
- API endpoint for serving and uploading preview images
- Frontend shows preview thumbnails in search results

---

## Phase 5: Collapsible Category Dropdowns
- Replaced flat category tabs with collapsible accordion sections
- Characters section expanded by default
- Each section shows LoRA count and expand/collapse arrow

---

## Phase 6: Video Job Partial Resumption
- `resume_interrupted_job()` loads last saved latent checkpoint
- `generate_video()` accepts resume params and injects saved latent
- Falls back to full re-run only if checkpoint loading fails

---

## Phase 7: Utility Script Relocation
- Moved `decode_checkpoint.py` and `_list_ckpts.py` to `scripts/` (neither was imported anywhere)

---

## Key Technical Details
- Preview image convention: `models/<category>/<lora_name>.preview.{png,jpg,jpeg,webp}`
- Multi-character gating: only LoRAs whose parent directory resolves to `models/characters/` count
- Resume latent: loaded with `weights_only=True`, moved to cuda/bfloat16 before injection

## Key Learnings
- Auto-inject trigger words on LoRA toggle — users forget to add them manually
- Don't strip user-specified weights from prompts — `(word:1.5)` is intentional
- Category gating by directory path is more reliable than keyword matching
- Checkpoint-based video resumption saves hours of GPU time on interrupted jobs
