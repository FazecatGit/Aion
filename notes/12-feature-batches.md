# Feature Batches — Changelog

---

## Feature Batch 2

### Backend
- **api.py**: Fixed two-mode agent endpoint (was passing wrong kwargs). Added 8 new endpoints: curriculum generation, TTS Kokoro, image generation, visualization modules
- **tutor/gamification.py**: Tiered XP scaling brackets: 1-30 (10%), 31-60 (15%), 61-90 (20%), 91+ (25%)
- **tutor/tutor.py**: MCQ deduplication, curriculum generation (`generate_full_chapter`, `get_generated_chapter`, `get_chapter_problem`), persists to `cache/generated_curriculum/`

### Frontend
- Removed Problem Bank button and panel (disabled feature)
- Full-width tutor mode: maxWidth → 98vw
- Agent mode dropdown: replaced separate toggle buttons with single `<select>`
- Expandable textarea with Shift+Enter for newlines
- Triangle viz → SOH CAH TOA right triangle with θ slider, labeled sides
- Vector viz → 2D/3D toggle, isometric 3D projection with cross product
- Math viz → tab-based navigation, close button
- Side chat expansion: wider chat container + input bar in agent mode
- Curriculum chapter generation with progress bar polling
- TTS button: 🔊 reads last AI message via Kokoro TTS endpoint

### Curriculum API Flow
1. POST `/curriculum/generate` — starts async generation
2. GET `/curriculum/generate/status` — poll for progress
3. GET `/curriculum/chapter/{subject}/{chapter}` — get generated chapter
4. GET `/curriculum/chapter/{subject}/{chapter}/{index}` — get specific problem

---

## Feature Batch 3

- **Compel fix**: `truncate_long_prompts=False`, fallback chain, `models/characters/` scanning
- **History refresh**: `fetchImageHistory()` in finally block of `handleImageGenerate()`
- **Orb activation**: `setMode('querying')`/`setMode('idle')` around image generation
- **Code agent speedup**: context 12K→8K, snippets 20K→12K, `num_predict=2048` for analysis
- **Feedback keywords**: 8 new issue keywords (long, short, hair, background, color, anatomy, feet, proportion)
- **Tutor UI**: Full-width panel, Math/CS toggle first, tools bar moved to top
- **CS curriculum**: 5 new subjects (Functional Programming, Web Dev, Testing, AI/ML, Compilers)
- **Data dropdown**: Consolidated 4 buttons into single "📂 Data ▾" dropdown
- **Character LoRA**: `list_models()` now scans `models/characters/` subfolder

---

## Feature Batch 4

### Bug Fixes
- **Generate tab animation bug**: `handleImageGenerate()` now accepts `animatedOverride?: boolean` to avoid stale React state

### Image Generation Improvements
- **Oil spill fix (LAB color)**: Rewrote `_match_color_to_reference()` with LAB color space transfer + brightness guard (±5% luminance clamp)

### GPU & Progress Monitoring
- `_generation_progress` global dict, `_update_progress()`, `get_generation_progress()`, `get_gpu_info()`, `_check_vram_safe()` (92% threshold)
- Auto-save on near-OOM: Animation frame loop auto-breaks at VRAM > 92%
- Frontend GPU bar: color-coded green/orange/red + step-level progress bar
- Polling: 1s during gen, 5s idle

### WAN Video Generation
- Backend: T2V/I2V, dual-LoRA (high_noise/low_noise), CPU offload
- Frontend: Video sub-tab with reference image, frames/fps/width/height sliders, action LoRA dropdown

### Smarter Body Movement Detection
- Expanded body parts (hips, chest, feet)
- Rich action map with aliases, steps, involved_parts, physics types
- Compound pose detection via `_POSE_IMPLICATIONS` dict
- Physics-based easing: ease-out cubic, ease-in cubic, sinusoidal, damping, bell curve
- Multi-action blending: primary drives motion, secondary adds hints
- Staggered idle animation, ambient movement for uncovered parts

## Key Learnings
- LAB color space is far superior to RGB for color transfer (no "oil spill" effect)
- Auto-save on near-OOM prevents losing long animation jobs
- Physics-based easing per action type creates much more natural motion
- Staggered idle animation prevents the "robot breathing" look
