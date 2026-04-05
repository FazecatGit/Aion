# Image Studio Overhaul

---

## Art Style System
- `ART_STYLES` dict with presets: anime, ghibli, manga, painterly, game_concept, realistic, custom
- Each has prompt prefix + style-specific negative to avoid "lifeless masterpiece"

---

## img2img Pipeline
- `_load_img2img_pipeline()` loads StableDiffusionXLImg2ImgPipeline
- Shares weights with txt2img via component injection (saves VRAM)

---

## Animation System
- `generate_animated()`: Full rewrite — uses img2img chaining for frame-to-frame consistency
  - `frame_strength` 0.35 default
  - Supports save/resume checkpoints per job_id
  - Storyboard per-frame descriptions
  - Art styles, LoRA stacking
  - GIF+MP4 output
- `regenerate_frame()`: Re-gen single frame blending neighbour frames as init_image
- `list_animation_jobs()`: Lists all saved animation savepoints
- `generate_storyboard()`: Low-res (256x256) 6-step sketch grid for quick previews

---

## Preview System
- `generate_with_preview()`: Two-pass generation (quarter-res preview → full render, same seed)

---

## Upscaling
- `upscale_image()`: Tiled upscaler with feathered blending, tile_size=768 for 8GB VRAM

---

## Training Pipeline
- `critique_training_dataset()`: Analyses image count, resolution, diversity, character coverage
- `start_lora_training()`: Threaded LoRA training using peft + diffusers
  - Gradient checkpointing
  - Resume from epoch checkpoints
  - Saves to `models/loras/<name>.safetensors`

---

## API Endpoints Added
- `/generate/animated/jobs`, `/generate/animated/frame/edit`
- `/generate/styles`, `/generate/storyboard`, `/generate/preview`, `/generate/upscale`
- `/generate/train/critique`, `/generate/train/start`, `/generate/train/status`, `/generate/train/cancel`

---

## Frontend
- ImageGen tab → ImageStudio with 5 sub-tabs: Generate, Animate, Storyboard, Upscale, Train
- Art style selector, frame strength slider, output format picker (gif/mp4/both)
- Storyboard descriptions textarea, animation job list panel
- Upscale tab: image path, scale slider, tile size
- Train tab: LoRA name, image dir, type (style/character), critique button, progress bar

## Key Learnings
- img2img chaining with shared pipeline weights saves ~50% VRAM vs separate pipelines
- Frame strength 0.35 is the sweet spot for animation consistency vs variety
- Checkpoint save/resume is essential for long animation jobs (VRAM can run out mid-job)
- Tiled upscaling with feathered blending avoids visible seams on 8GB VRAM
