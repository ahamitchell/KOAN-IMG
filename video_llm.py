"""video_llm.py — Claude Haiku prompt enhancer for KOAN.img VIDEO tab.

Given a start frame image (+ optional end frame), a user hint, and a style preference,
asks Claude Haiku 3.5 to write an optimised image-to-video prompt for the target model.

Network call is synchronous — run inside a QThread.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Optional

# ── constants ──────────────────────────────────────────────────────────────────
LLM_MODEL  = "claude-3-haiku-20240307"
MAX_TOKENS = 300   # prompts are short; keep cost minimal

STYLE_GUIDE: dict[str, str] = {
    "literal": (
        "Write a concise, literal motion description (≤ 60 words). "
        "Describe realistic camera movement, physical actions, and natural transitions "
        "that make sense between the two frames. Be specific and grounded. "
        "No poetic flourishes."
    ),
    "abstract": (
        "Write an expressive, cinematic prompt (≤ 60 words). "
        "Use sensory, atmospheric language. Prioritise mood, colour, rhythm, and emotion "
        "over literal description. Be evocative and visually rich."
    ),
}

MODEL_NOTES: dict[str, str] = {
    "Seedance 1.0 Pro": (
        "Seedance 1.0 Pro prefers clear subject-focus prompts. Keep it simple and direct."
    ),
    "Kling 2.1 Pro": (
        "Kling 2.1 Pro handles creative and high-motion content. "
        "Cinematic framing language works well."
    ),
    "Kling 1.6 Pro": (
        "Kling 1.6 Pro is reliable for clean motion and subject tracking. "
        "Keep prompts direct and action-focused."
    ),
    "Kling 3.0 Pro": (
        "Kling 3.0 Pro is the latest generation — excellent motion quality and coherence. "
        "Handles complex actions and camera movement well. Cinematic language encouraged."
    ),
    "Higgsfield DoP": (
        "Higgsfield DoP is a cinematographer model. Emphasise camera movement, "
        "lens language, lighting quality, and depth of field."
    ),
}

_SYSTEM = (
    "You are a video-generation prompt engineer. "
    "The user will show you one or two key frames and an optional hint. "
    "Your job: write a concise, effective image-to-video prompt.\n\n"
    "Rules:\n"
    "• Output ONLY the prompt text — no preamble, no quotes, no explanation.\n"
    "• Maximum 70 words.\n"
    "• Structure: subject/scene → motion/action → mood/style.\n"
    "• Avoid clichés: 'seamlessly', 'magical', 'breathtaking', 'beautiful'.\n"
    "• Never describe the frame as a photo or image; treat it as a real scene.\n"
)


# ── helpers ────────────────────────────────────────────────────────────────────
def _load_api_key() -> str:
    cfg_path = Path(__file__).parent / "koan_config.json"
    if not cfg_path.exists():
        raise RuntimeError("koan_config.json not found.")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    key = cfg.get("anthropic_api_key", "").strip()
    if not key:
        raise RuntimeError(
            "Anthropic key is empty — open ⚙ API Keys in the VIDEO tab and paste your sk-ant-… key."
        )
    if not key.startswith("sk-ant-"):
        raise RuntimeError(
            f"Anthropic key looks wrong (starts with '{key[:8]}…'). "
            "It should start with  sk-ant-  — check ⚙ API Keys."
        )
    return key


def _to_jpeg_bytes(path: str) -> bytes:
    """Convert any image (including WebP) to JPEG bytes in memory."""
    from PIL import Image as _PILImage
    import io
    img = _PILImage.open(path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def _encode_image(path: str) -> tuple[str, str]:
    """Return (base64_data, media_type) for a local image file.

    WebP and other non-JPEG/PNG formats are converted to JPEG first
    for maximum API compatibility.
    """
    ext = Path(path).suffix.lower()
    if ext in (".jpg", ".jpeg", ".png"):
        media_type = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
        data = base64.standard_b64encode(Path(path).read_bytes()).decode("utf-8")
    else:
        # WebP, GIF, BMP, TIFF, etc. → convert to JPEG
        media_type = "image/jpeg"
        data = base64.standard_b64encode(_to_jpeg_bytes(path)).decode("utf-8")
    return data, media_type


# ── public API ─────────────────────────────────────────────────────────────────
def enhance_prompt(
    first_frame:  str,
    last_frame:   Optional[str]  = None,
    hint:         str            = "",
    style:        str            = "literal",   # "literal" | "abstract"
    model_name:   str            = "Seedance 1.0 Pro",
    llm_model:    str            = LLM_MODEL,
    global_style: str            = "",
) -> str:
    """Call Claude Haiku to generate an image-to-video prompt.

    Parameters
    ----------
    first_frame : local path to the start-frame image
    last_frame  : local path to the end-frame image, or None
    hint        : free-text hint from the user (may be empty)
    style       : "literal" or "abstract"
    model_name  : target Higgsfield model (used to tailor the prompt)
    llm_model   : which Claude model to use

    Returns
    -------
    str — the generated prompt text (stripped)
    """
    import anthropic  # lazy import — module usable even without anthropic installed

    api_key = _load_api_key()
    client  = anthropic.Anthropic(api_key=api_key)

    style_instruction = STYLE_GUIDE.get(style, STYLE_GUIDE["literal"])
    model_note        = MODEL_NOTES.get(model_name, "")

    # ── assemble multimodal user message ─────────────────────────────────────
    content: list = []

    # Start frame
    data, mt = _encode_image(first_frame)
    content.append({"type": "image",
                    "source": {"type": "base64", "media_type": mt, "data": data}})
    content.append({"type": "text", "text": "This is the START frame."})

    # End frame (optional)
    if last_frame:
        data2, mt2 = _encode_image(last_frame)
        content.append({"type": "image",
                        "source": {"type": "base64", "media_type": mt2, "data": data2}})
        content.append({"type": "text", "text": "This is the END frame."})

    # Instructions block
    user_text = style_instruction
    if model_note:
        user_text += f"\n\nTarget model: {model_note}"
    if global_style.strip():
        user_text += (
            f"\n\nGlobal visual style (MUST be reflected throughout — treat this as "
            f"the primary creative direction for every prompt you write): "
            f"{global_style.strip()}"
        )
    if hint.strip():
        user_text += f"\n\nPer-clip concept hint: {hint.strip()}"
    user_text += "\n\nNow write the prompt:"
    content.append({"type": "text", "text": user_text})

    # ── call API ──────────────────────────────────────────────────────────────
    resp = client.messages.create(
        model      = llm_model,
        max_tokens = MAX_TOKENS,
        system     = _SYSTEM,
        messages   = [{"role": "user", "content": content}],
    )

    return resp.content[0].text.strip()
