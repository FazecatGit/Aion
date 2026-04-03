"""Art style presets and application."""

ART_STYLES: dict[str, dict] = {
    "anime": {
        "prefix": (
            "anime screencap, hand-drawn animation, cel shading, "
            "soft lighting, natural colour palette, expressive linework, "
            "high quality anime key visual"
        ),
        "negative_extra": (
            "3d render, photorealistic, CGI, plastic, airbrushed, "
            "overly saturated, neon colours, smooth shading, "
            "deviantart, artstation trending, western comic"
        ),
    },
    "ghibli": {
        "prefix": (
            "studio ghibli style, watercolour background, "
            "warm lighting, soft pastel tones, hand-painted, "
            "detailed environment, gentle linework"
        ),
        "negative_extra": (
            "3d render, photorealistic, digital art, "
            "overly saturated, neon, sharp lines, CGI"
        ),
    },
    "manga": {
        "prefix": (
            "manga panel, black and white, screentone, "
            "detailed linework, dramatic shading, ink drawing"
        ),
        "negative_extra": (
            "colour, colorful, 3d render, photorealistic, "
            "painting, watercolour"
        ),
    },
    "painterly": {
        "prefix": (
            "oil painting, visible brush strokes, impressionist lighting, "
            "textured canvas, rich earth tones, gallery quality"
        ),
        "negative_extra": (
            "anime, cartoon, digital art, smooth shading, "
            "flat colours, 3d render, photography"
        ),
    },
    "game_concept": {
        "prefix": (
            "game concept art, splash art, dynamic composition, "
            "dramatic rim lighting, painterly rendering, "
            "detailed character design"
        ),
        "negative_extra": (
            "photograph, photorealistic, anime, flat colours, "
            "simple background, sketch"
        ),
    },
    "realistic": {
        "prefix": (
            "photorealistic, RAW photo, 8k UHD, DSLR, "
            "professional lighting, studio quality"
        ),
        "negative_extra": (
            "anime, cartoon, drawing, painting, illustration, "
            "stylized, flat colours"
        ),
    },
    "3d_blender": {
        "prefix": (
            "3d render, blender, unreal engine 5, octane render, "
            "volumetric lighting, subsurface scattering, PBR materials, "
            "video game character, detailed textures, sharp focus"
        ),
        "negative_extra": (
            "anime, hand-drawn, sketch, painting, flat colours, "
            "2d, cel shading, watercolour, pencil, photo"
        ),
    },
    "custom": {
        "prefix": "",
        "negative_extra": "",
    },
}

STYLE_NAMES = list(ART_STYLES.keys())


def get_art_styles() -> list[dict]:
    """Return available art style presets for the frontend."""
    return [
        {"name": k, "prefix": v["prefix"][:80] + "…" if len(v["prefix"]) > 80 else v["prefix"]}
        for k, v in ART_STYLES.items()
    ]


def _apply_art_style(prompt: str, negative: str, style: str) -> tuple[str, str]:
    """Prepend style prefix and extend negative with style-specific exclusions."""
    preset = ART_STYLES.get(style)
    if not preset or style == "custom":
        return prompt, negative
    styled_prompt = f"{preset['prefix']}, {prompt}" if preset["prefix"] else prompt
    styled_neg = f"{negative}, {preset['negative_extra']}" if preset["negative_extra"] else negative
    return styled_prompt, styled_neg
