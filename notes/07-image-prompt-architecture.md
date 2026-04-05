# Image Prompt Building Architecture

---

## Overview
Multi-character image generation uses a sophisticated system combining LoRA trigger words, outfit groups, BREAK token chunking, and automatic prompt analysis for the UI display.

---

## 1. Trigger Word System
- **File**: `models/trigger_words.json`
- **Parsed by**: `agent/image_generation.py`
  - `parse_trigger_word_entry()` — Handles dict/string parsing with weights
  - `format_trigger_word()` — Formats with weight syntax (word:1.2)
  - `parse_outfit_groups()` — Splits by ";" for outfit groups
  - `build_trigger_prompt()` — Builds final trigger string

---

## 2. Multi-Character LoRA Handling
- **Function**: `_structure_multi_character_prompt()`
  - Only activates with 2+ character-category LoRAs
  - Each character gets position ("left", "right", "center", "background")
  - Characters separated by " BREAK " tokens
  - Trigger words injected per character using outfit-aware system
  - Base words removed from shared prompt

---

## 3. BREAK Token Chunking
- `_chunk_prompt()` — Splits at 75-token boundaries
- `_encode_long_prompt()` — Uses Compel for long prompts
- `_manual_chunk_encode()` — Fallback chunking
- Joins chunks with " BREAK " to bypass 77-token CLIP limit

---

## 4. Prompt Analysis Display (UI)
- Renders `estimated_focus` with colored tags
  - Green/"high" (positions 1-3)
  - Orange/"medium" (4-8)
  - Gray/"low" (9+)
- Shows seed, model, long_prompt flag
- Collapsible sections for LoRA status, injected triggers, full prompt

---

## 5. Auto-Injection Logic
- Checks if trigger words missing from prompt
- Auto-injects at start with outfit awareness
- Logs injected triggers and `lora_diagnostics`
- Returned in API response headers

---

## API Endpoints
- **POST /generate/image** — Main generation with selected_outfits
- **GET /generate/history** — Returns history with prompt_analysis
- Response headers include X-Generation-Meta with all analysis data

## Key Learnings
- BREAK tokens are essential for multi-character isolation in SDXL
- Auto-injection of missing trigger words prevents user error
- Prompt analysis with colored priority tags helps users understand what the model focuses on
