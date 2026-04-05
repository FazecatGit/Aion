# Animation & Image Quality Improvements

---

## Multi-Angle Prompts
- `_structure_multi_angle_prompt()` detects "side view and front view" patterns
- Restructures as split-screen BREAK-separated composition

---

## Multi-Character Prompts
- `_structure_multi_character_prompt()` scans trigger_words.json for character names
- Positions them (left/right/center) with BREAK separation

---

## Preview Consistency Fix
- `generate_preview_only()` now uses FULL resolution + same seed + fewer steps (4-8), then downscales
- Produces noisy preview, not a different image

---

## Animation Deterioration Fix
- `_match_color_to_reference()` per-channel mean/std color transfer anchors each frame to frame 0's histogram
- Reference blending: Each frame blends with frame 0 at `reference_blend` ratio (default 0.3)
- Prevents cumulative drift

---

## Strength Curves
- `_build_strength_curve()` supports "constant", "ease_in", "pulse" denoising strength per frame

---

## Auto Body Movement Storyboard
- `_auto_generate_motion_storyboard()` parses prompt for body parts/actions
- Uses `_BODY_PART_ACTIONS` and `_ACTION_MOTION_MAP` dicts

---

## Save Animation State
- `save_animation_state()`: New function to save animation form state as a named job without generating

---

## API Changes
- `AnimatedGenRequest` has 3 new fields: `reference_blend`, `strength_curve`, `motion_intensity`
- New `SaveAnimStateRequest` model + `POST /generate/animated/save` endpoint

---

## Frontend Changes
- State vars: `animReferenceBlend`, `animStrengthCurve`, `animMotionIntensity`, `modelsExpanded`
- Animate tab: Reference Blend slider (0-0.6), Motion Intensity slider (0-1), Strength Curve selector
- Save State button → calls `/generate/animated/save`, then refreshes job list
- Models panel converted to collapsible dropdown button

## Key Learnings
- LAB color space transfer prevents the "oil spill" effect in frame-to-frame animation
- Reference blending at 0.3 is the sweet spot — prevents drift without making frames identical
- Strength curves add natural motion feel — "ease_in" for acceleration, "pulse" for impacts
