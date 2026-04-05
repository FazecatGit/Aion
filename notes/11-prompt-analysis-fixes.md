# Prompt Analysis & Character Detection Fixes — April 2026

---

## Backend Changes (`agent/image_generation.py`)

### Improved `_analyze_prompt`
- Now detects character trigger codes in prompt text and marks them as `priority: "character"` with `section: "character_inline"`
- Prevents character tags from being labeled as "low priority shared" even when they appear before any BREAK token

### Auto-BREAK Insertion
- When a prompt has no BREAK tokens but contains parenthesized blocks with known character trigger codes, BREAKs are auto-inserted before each block
- Applied in both `generate_image` and `generate_preview_only`

### 29/29 Completion Display
- `finally` block in `generate_image` keeps `active=True` with "Complete!" for 3 seconds via a daemon thread
- Frontend poll catches completion state

### `<lora:...>` Tag Stripping
- A1111/ComfyUI-style lora tags are stripped from prompt text (this backend loads LoRAs via Python API)

### Character Validation (`validate_prompt_characters`)
- Fuzzy-matches user-typed names against known trigger codes
- API endpoint at `/generate/validate-prompt`

### History Deletion
- `delete_generation_history_entry` API endpoint at `DELETE /generate/history/{index}`

---

## Frontend Changes (`renderer.tsx`)

### Character Validation UI
- Auto-validates prompt against known trigger codes with debounced API calls
- Green checkmarks for matched characters
- Orange warnings for misspellings with suggestions

### Prompt Analysis Improvements
- Shows `character_inline` sections with orange warning label ("add BREAK for better isolation")
- Shows BREAK segment count

### History Expandable/Removable
- Click to expand with full details (settings, analysis tags, load/open/upscale/delete buttons)
- Collapsed view has quick action buttons including delete
- Only one item expanded at a time

### Sticky Image Tile
- Result area: `position: sticky, top: 0, flexShrink: 0` with `height: 450px`
- Outer center panel `overflow: hidden` removed to allow sticky to work

### Progress Completion
- Shows "✓ Complete!" with green border when generation finishes

---

## Key Design Decisions
- Auto-BREAK only triggers when the prompt has NO BREAK tokens at all (avoids double-inserting)
- Character detection uses exact substring matching against known trigger codes
- Fuzzy matching uses character-level LCS ratio with 0.6 threshold for misspelling detection
- Preview uses 256×256 resolution for speed

## Key Learnings
- Auto-BREAK insertion dramatically improves multi-character quality for users who forget
- Fuzzy matching with 0.6 threshold catches most misspellings without false positives
- Sticky image tile keeps the result visible while scrolling through prompt analysis
