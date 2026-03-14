"""video_api.py — Higgsfield API client for KOAN.img VIDEO tab.

Uses the official higgsfield_client SDK (SyncClient).
All network calls are synchronous — run inside a QThread.

Auth: SyncClient(api_key="{key}:{secret}")  →  Authorization: Key {key}:{secret}
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Optional

# ── MIME type map (Python's mimetypes misses webp on Windows) ─────────────────
_MIME: dict[str, str] = {
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png":  "image/png",
    ".gif":  "image/gif",
    ".webp": "image/webp",
    ".bmp":  "image/bmp",
    ".tiff": "image/tiff",
    ".tif":  "image/tiff",
}

def _to_jpeg_bytes(path: str) -> bytes:
    """Convert any image (including WebP) to JPEG bytes in memory."""
    from PIL import Image as _PILImage
    import io
    img = _PILImage.open(path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def _upload(client, path: str) -> str:
    """Upload a local image with an explicitly resolved MIME type.

    WebP and other non-JPEG/PNG formats are converted to JPEG first
    for maximum API compatibility.
    """
    p = Path(path)
    ext = p.suffix.lower()
    if ext in (".jpg", ".jpeg", ".png"):
        mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
        data = p.read_bytes()
    else:
        # WebP, GIF, BMP, TIFF, etc. → convert to JPEG
        mime = "image/jpeg"
        data = _to_jpeg_bytes(path)
    return client.upload(data, mime)

# ── constants ─────────────────────────────────────────────────────────────────
POLL_INTERVAL = 6    # seconds between status checks (fallback)
MAX_WAIT      = 600  # seconds before timeout

# ── model registry ────────────────────────────────────────────────────────────
# resolution values are display strings (e.g. "720p").
# generate() converts them to the per-model API format automatically.
MODELS = {
    "Seedance 1.0 Pro": {
        "id":            "bytedance/seedance/v1/pro/image-to-video",
        "has_audio":     False,
        "has_end_frame": True,
        "has_camera_fixed": False,
        "duration_min":  5,
        "duration_max":  10,
        "duration_step": 5,
        "resolutions":   ["480p", "720p", "1080p"],
        "aspect_ratios": ["16:9", "9:16", "1:1", "4:3", "3:4", "21:9"],
    },
    "Kling 2.1 Pro": {
        "id":            "kling-video/v2.1/pro/image-to-video",
        "has_audio":     False,
        "has_end_frame": True,
        "has_camera_fixed": False,
        "duration_min":  5,
        "duration_max":  10,
        "duration_step": 5,
        "resolutions":   ["720p", "1080p"],
        "aspect_ratios": ["16:9", "9:16", "1:1"],
    },
    "Higgsfield DoP": {
        "id":            "higgsfield-ai/dop/standard",
        "has_audio":     False,
        "has_end_frame": False,
        "has_camera_fixed": False,
        "duration_min":  5,
        "duration_max":  5,
        "duration_step": 1,
        "resolutions":   ["720p"],
        "aspect_ratios": ["16:9", "9:16", "1:1"],
    },
    "Kling 1.6 Pro": {
        "id":            "kling-video/v1.6/pro/image-to-video",
        "has_audio":     False,
        "has_end_frame": True,
        "has_camera_fixed": False,
        "duration_min":  5,
        "duration_max":  10,
        "duration_step": 5,
        "resolutions":   ["720p", "1080p"],
        "aspect_ratios": ["16:9", "9:16", "1:1"],
    },
    "Kling 3.0 Pro": {
        "id":            "kling-video/v3.0/pro/image-to-video",
        "has_audio":     False,
        "has_end_frame": True,
        "has_camera_fixed": False,
        "duration_min":  5,
        "duration_max":  10,
        "duration_step": 5,
        "resolutions":   ["720p", "1080p"],
        "aspect_ratios": ["16:9", "9:16", "1:1"],
    },
}


class HiggsfieldError(RuntimeError):
    pass


class HiggsfieldClient:
    def __init__(self, api_key: str, api_secret: str):
        self._api_key    = api_key.strip()
        self._api_secret = api_secret.strip()

    def _auth_token(self) -> str:
        """Combined token: key:secret  (as confirmed by diagnostics)."""
        if self._api_secret:
            return f"{self._api_key}:{self._api_secret}"
        return self._api_key

    def _client(self):
        """Return a fresh SyncClient.  api_key is passed directly — no env vars."""
        import higgsfield_client as hfc
        return hfc.SyncClient(api_key=self._auth_token())

    # ── upload ────────────────────────────────────────────────────────────────
    def upload_image(self, path: str) -> str:
        """Upload a local image/video frame and return the hosted URL."""
        return _upload(self._client(), path)

    # ── generate (one-shot) ───────────────────────────────────────────────────
    def generate(
        self,
        model_id:     str,
        first_frame:  str,
        last_frame:   Optional[str],
        prompt:       str,
        dest_path:    str,
        progress_cb:  Optional[Callable[[str], None]] = None,
        cancel_check: Optional[Callable[[], bool]]    = None,
        **kwargs,
    ) -> str:
        """Upload frames → submit → wait → download.  Returns dest_path."""
        import higgsfield_client as hfc

        client = self._client()
        model_info = next((m for m in MODELS.values() if m["id"] == model_id), {})

        # ── upload frames ─────────────────────────────────────────────────────
        if progress_cb:
            progress_cb("Uploading first frame…")
        first_url = _upload(client, first_frame)

        last_url: Optional[str] = None
        if last_frame and model_info.get("has_end_frame"):
            if progress_cb:
                progress_cb("Uploading last frame…")
            last_url = _upload(client, last_frame)

        # ── resolution: Seedance API uses '720' not '720p' ────────────────────
        resolution_display = kwargs.get("resolution", "720p")
        if model_id.startswith("bytedance/seedance"):
            resolution_api = resolution_display.rstrip("p")   # "720p" → "720"
        else:
            resolution_api = resolution_display               # Kling/DoP keep "720p"

        # ── duration: always int ──────────────────────────────────────────────
        try:
            duration_int = int(kwargs.get("duration", 5))
        except (TypeError, ValueError):
            duration_int = 5

        # ── build arguments ───────────────────────────────────────────────────
        arguments: dict = {
            "prompt":       prompt,
            "image_url":    first_url,
            "aspect_ratio": kwargs.get("aspect_ratio", "16:9"),
            "resolution":   resolution_api,
            "duration":     duration_int,
        }

        if last_url:
            arguments["end_image_url"] = last_url

        seed = kwargs.get("seed", -1)
        if seed not in (-1, None):
            arguments["seed"] = int(seed)

        if model_info.get("has_audio") and kwargs.get("generate_audio") is not None:
            arguments["generate_audio"] = bool(kwargs["generate_audio"])

        if model_info.get("has_camera_fixed") and kwargs.get("camera_fixed") is not None:
            arguments["camera_fixed"] = bool(kwargs["camera_fixed"])

        # ── submit + poll ─────────────────────────────────────────────────────
        if progress_cb:
            progress_cb("Submitting job…")

        def _on_update(status):
            if cancel_check and cancel_check():
                return
            name = type(status).__name__
            if progress_cb:
                progress_cb(name)

        try:
            result = client.subscribe(
                model_id,
                arguments,
                on_queue_update=_on_update,
            )
        except Exception as exc:
            if cancel_check and cancel_check():
                raise HiggsfieldError("Cancelled by user.")
            raise HiggsfieldError(str(exc)) from exc

        # ── check for content moderation rejection ────────────────────────────
        if isinstance(result, dict) and result.get("status") == "nsfw":
            raise HiggsfieldError(
                "NSFW — Higgsfield's content filter rejected this clip. "
                "Revise the prompt: avoid graphic gore, explicit body horror, or disturbing imagery."
            )

        # ── extract video URL ─────────────────────────────────────────────────
        video_url = _extract_video_url(result)
        if not video_url:
            raise HiggsfieldError(f"No video URL in result: {str(result)[:300]}")

        if progress_cb:
            progress_cb("Downloading…")
        return _download(video_url, dest_path)


# ── helpers ───────────────────────────────────────────────────────────────────

def _extract_video_url(result) -> Optional[str]:
    """Try several common result shapes to find the video URL."""
    if not result:
        return None
    if isinstance(result, str) and result.startswith("http"):
        return result
    if isinstance(result, dict):
        v = result.get("video")
        if isinstance(v, dict):
            return v.get("url")
        for key in ("video_url", "url", "output", "output_url"):
            val = result.get(key)
            if isinstance(val, str) and val.startswith("http"):
                return val
        outputs = result.get("outputs") or result.get("output")
        if isinstance(outputs, list) and outputs:
            if isinstance(outputs[0], str):
                return outputs[0]
    return None


def _download(url: str, dest_path: str) -> str:
    import requests
    Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, timeout=120, stream=True)
    resp.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=65536):
            f.write(chunk)
    return dest_path


def load_client() -> HiggsfieldClient:
    """Load credentials from koan_config.json."""
    cfg_path = Path(__file__).parent / "koan_config.json"
    if not cfg_path.exists():
        raise HiggsfieldError(
            "koan_config.json not found. Add your API keys via ⚙ API Keys."
        )
    cfg    = json.loads(cfg_path.read_text(encoding="utf-8"))
    key    = cfg.get("higgsfield_api_key",    "").strip()
    secret = cfg.get("higgsfield_api_secret", "").strip()
    if not key:
        raise HiggsfieldError(
            "higgsfield_api_key not set. Open ⚙ API Keys in the VIDEO tab."
        )
    return HiggsfieldClient(key, secret)
