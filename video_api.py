"""video_api.py — Video generation API clients for KOAN.img VIDEO tab.

Supports three providers:
  1. Higgsfield  — via higgsfield_client SDK (SyncClient)
  2. fal.ai      — via fal_client SDK (subscribe)
  3. Kling       — via Kling AI REST API (api.klingai.com) with JWT auth

All network calls are synchronous — run inside a QThread.
"""
from __future__ import annotations

import base64
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
        mime = "image/jpeg"
        data = _to_jpeg_bytes(path)
    return client.upload(data, mime)

# ── constants ─────────────────────────────────────────────────────────────────
POLL_INTERVAL = 5    # seconds between status checks
MAX_WAIT      = 600  # seconds before timeout

# ── model registry ────────────────────────────────────────────────────────────
# "provider" is "higgsfield", "fal", or "kling".
MODELS = {
    # ── Higgsfield ────────────────────────────────────────────────────────────
    "Kling 2.1 Pro  (Higgsfield)": {
        "id":            "kling-v2.1-pro-i2v",
        "provider":      "higgsfield",
        "has_audio":     False,
        "has_end_frame": True,
        "has_camera_fixed": False,
        "duration_min":  5,
        "duration_max":  10,
        "duration_step": 5,
        "resolutions":   ["720p", "1080p"],
        "aspect_ratios": ["16:9", "9:16", "1:1"],
    },
    "Kling 2.5 Pro  (Higgsfield)": {
        "id":            "kling-v2.5-pro-i2v",
        "provider":      "higgsfield",
        "has_audio":     False,
        "has_end_frame": True,
        "has_camera_fixed": False,
        "duration_min":  5,
        "duration_max":  10,
        "duration_step": 5,
        "resolutions":   ["720p", "1080p"],
        "aspect_ratios": ["16:9", "9:16", "1:1"],
    },
    "Kling 2.6 Pro  (Higgsfield)": {
        "id":            "kling-v2.6-pro-i2v",
        "provider":      "higgsfield",
        "has_audio":     False,
        "has_end_frame": True,
        "has_camera_fixed": False,
        "duration_min":  5,
        "duration_max":  10,
        "duration_step": 5,
        "resolutions":   ["720p", "1080p"],
        "aspect_ratios": ["16:9", "9:16", "1:1"],
    },
    "Seedance Pro  (Higgsfield)": {
        "id":            "seedance-pro-i2v",
        "provider":      "higgsfield",
        "has_audio":     False,
        "has_end_frame": True,
        "has_camera_fixed": False,
        "duration_min":  5,
        "duration_max":  10,
        "duration_step": 5,
        "resolutions":   ["480p", "720p", "1080p"],
        "aspect_ratios": ["16:9", "9:16", "1:1", "4:3", "3:4", "21:9"],
    },
    "Seedance 2.0  (Higgsfield)": {
        "id":            "seedance-v2.0-i2v",
        "provider":      "higgsfield",
        "has_audio":     True,
        "has_end_frame": True,
        "has_camera_fixed": False,
        "duration_min":  5,
        "duration_max":  12,
        "duration_step": 1,
        "resolutions":   ["480p", "720p", "1080p"],
        "aspect_ratios": ["16:9", "9:16", "1:1", "4:3", "3:4", "21:9"],
    },
    "Seedance Lite  (Higgsfield)": {
        "id":            "seedance-lite-i2v",
        "provider":      "higgsfield",
        "has_audio":     False,
        "has_end_frame": False,
        "has_camera_fixed": False,
        "duration_min":  5,
        "duration_max":  5,
        "duration_step": 1,
        "resolutions":   ["480p", "720p"],
        "aspect_ratios": ["16:9", "9:16", "1:1"],
    },
    "Wan 2.5  (Higgsfield)": {
        "id":            "wan2.5-i2v",
        "provider":      "higgsfield",
        "has_audio":     False,
        "has_end_frame": False,
        "has_camera_fixed": False,
        "duration_min":  5,
        "duration_max":  5,
        "duration_step": 1,
        "resolutions":   ["720p"],
        "aspect_ratios": ["16:9", "9:16", "1:1"],
    },
    "MiniMax Hailuo 02 Pro  (Higgsfield)": {
        "id":            "minimax-hailuo-02-pro-i2v",
        "provider":      "higgsfield",
        "has_audio":     False,
        "has_end_frame": False,
        "has_camera_fixed": False,
        "duration_min":  5,
        "duration_max":  5,
        "duration_step": 1,
        "resolutions":   ["720p", "1080p"],
        "aspect_ratios": ["16:9", "9:16", "1:1"],
    },
    "Higgsfield DoP  (Higgsfield)": {
        "id":            "higgsfield-dop-image-to-video",
        "provider":      "higgsfield",
        "has_audio":     False,
        "has_end_frame": False,
        "has_camera_fixed": False,
        "duration_min":  5,
        "duration_max":  5,
        "duration_step": 1,
        "resolutions":   ["720p"],
        "aspect_ratios": ["16:9", "9:16", "1:1"],
    },
    # ── fal.ai ────────────────────────────────────────────────────────────────
    "Seedance 1.0 Pro  (fal)": {
        "id":            "fal-ai/bytedance/seedance/v1/pro/image-to-video",
        "provider":      "fal",
        "has_audio":     False,
        "has_end_frame": True,
        "has_camera_fixed": True,
        "duration_min":  2,
        "duration_max":  12,
        "duration_step": 1,
        "resolutions":   ["480p", "720p", "1080p"],
        "aspect_ratios": ["16:9", "9:16", "1:1", "4:3", "3:4", "21:9"],
    },
    "Seedance 1.5 Pro  (fal)": {
        "id":            "fal-ai/bytedance/seedance/v1.5/pro/image-to-video",
        "provider":      "fal",
        "has_audio":     True,
        "has_end_frame": True,
        "has_camera_fixed": True,
        "duration_min":  4,
        "duration_max":  12,
        "duration_step": 1,
        "resolutions":   ["480p", "720p", "1080p"],
        "aspect_ratios": ["16:9", "9:16", "1:1", "4:3", "3:4", "21:9"],
    },
    "Kling 2.1 Pro  (fal)": {
        "id":            "fal-ai/kling-video/v2.1/pro/image-to-video",
        "provider":      "fal",
        "has_audio":     False,
        "has_end_frame": True,
        "has_camera_fixed": False,
        "duration_min":  5,
        "duration_max":  10,
        "duration_step": 5,
        "resolutions":   ["720p", "1080p"],
        "aspect_ratios": ["16:9", "9:16", "1:1"],
    },
    "Kling 2.6 Pro  (fal)": {
        "id":            "fal-ai/kling-video/v2.6/pro/image-to-video",
        "provider":      "fal",
        "has_audio":     False,
        "has_end_frame": True,
        "has_camera_fixed": False,
        "duration_min":  5,
        "duration_max":  10,
        "duration_step": 5,
        "resolutions":   ["720p", "1080p"],
        "aspect_ratios": ["16:9", "9:16", "1:1"],
    },
    "Kling 3.0 Pro  (fal)": {
        "id":            "fal-ai/kling-video/v3/pro/image-to-video",
        "provider":      "fal",
        "has_audio":     True,
        "has_end_frame": True,
        "has_camera_fixed": False,
        "duration_min":  3,
        "duration_max":  15,
        "duration_step": 1,
        "resolutions":   ["720p", "1080p"],
        "aspect_ratios": ["16:9", "9:16", "1:1"],
    },
    "MiniMax Hailuo  (fal)": {
        "id":            "fal-ai/minimax-video/image-to-video",
        "provider":      "fal",
        "has_audio":     False,
        "has_end_frame": False,
        "has_camera_fixed": False,
        "duration_min":  5,
        "duration_max":  5,
        "duration_step": 1,
        "resolutions":   ["720p"],
        "aspect_ratios": ["16:9", "9:16", "1:1"],
    },
    # ── Kling Direct (api.klingai.com) ────────────────────────────────────────
    "Kling 2.1 Pro  (Kling)": {
        "id":            "kling-v2-1",
        "provider":      "kling",
        "kling_mode":    "pro",
        "has_audio":     False,
        "has_end_frame": True,
        "has_camera_fixed": False,
        "duration_min":  5,
        "duration_max":  10,
        "duration_step": 5,
        "resolutions":   ["720p", "1080p"],
        "aspect_ratios": ["16:9", "9:16", "1:1"],
    },
    "Kling 2.1 Std  (Kling)": {
        "id":            "kling-v2-1",
        "provider":      "kling",
        "kling_mode":    "std",
        "has_audio":     False,
        "has_end_frame": True,
        "has_camera_fixed": False,
        "duration_min":  5,
        "duration_max":  10,
        "duration_step": 5,
        "resolutions":   ["720p", "1080p"],
        "aspect_ratios": ["16:9", "9:16", "1:1"],
    },
    "Kling 2.6 Pro  (Kling)": {
        "id":            "kling-v2-6",
        "provider":      "kling",
        "kling_mode":    "pro",
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

class FalError(RuntimeError):
    pass

class KlingError(RuntimeError):
    pass


# ═══════════════════════════════════════════════════════════════════════════════
#  Higgsfield client
# ═══════════════════════════════════════════════════════════════════════════════

class HiggsfieldClient:
    provider = "higgsfield"

    def __init__(self, api_key: str, api_secret: str):
        self._api_key    = api_key.strip()
        self._api_secret = api_secret.strip()

    def _auth_token(self) -> str:
        if self._api_secret:
            return f"{self._api_key}:{self._api_secret}"
        return self._api_key

    def _client(self):
        import higgsfield_client as hfc
        return hfc.SyncClient(api_key=self._auth_token())

    def upload_image(self, path: str) -> str:
        return _upload(self._client(), path)

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
        client = self._client()
        model_info = next((m for m in MODELS.values() if m["id"] == model_id), {})

        if progress_cb:
            progress_cb("Uploading first frame…")
        first_url = _upload(client, first_frame)

        last_url: Optional[str] = None
        if last_frame and model_info.get("has_end_frame"):
            if progress_cb:
                progress_cb("Uploading last frame…")
            last_url = _upload(client, last_frame)

        resolution_display = kwargs.get("resolution", "720p")
        if "seedance" in model_id and "v2" not in model_id:
            resolution_api = resolution_display.rstrip("p")
        else:
            resolution_api = resolution_display

        try:
            duration_int = int(kwargs.get("duration", 5))
        except (TypeError, ValueError):
            duration_int = 5

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

        if progress_cb:
            progress_cb("Submitting job…")

        def _on_update(status):
            if cancel_check and cancel_check():
                return
            if progress_cb:
                progress_cb(type(status).__name__)

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

        if isinstance(result, dict) and result.get("status") == "nsfw":
            raise HiggsfieldError(
                "NSFW — content filter rejected this clip. "
                "Revise the prompt."
            )

        video_url = _extract_video_url(result)
        if not video_url:
            raise HiggsfieldError(f"No video URL in result: {str(result)[:300]}")

        if progress_cb:
            progress_cb("Downloading…")
        return _download(video_url, dest_path)


# ═══════════════════════════════════════════════════════════════════════════════
#  fal.ai client
# ═══════════════════════════════════════════════════════════════════════════════

class FalClient:
    provider = "fal"

    def __init__(self, api_key: str):
        self._api_key = api_key.strip()

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
        import os
        os.environ["FAL_KEY"] = self._api_key
        import fal_client

        model_info = next((m for m in MODELS.values() if m["id"] == model_id), {})

        if progress_cb:
            progress_cb("Uploading first frame…")
        first_url = fal_client.upload_file(first_frame)

        last_url: Optional[str] = None
        if last_frame and model_info.get("has_end_frame"):
            if progress_cb:
                progress_cb("Uploading last frame…")
            last_url = fal_client.upload_file(last_frame)

        try:
            duration_val = str(int(kwargs.get("duration", 5)))
        except (TypeError, ValueError):
            duration_val = "5"

        resolution_display = kwargs.get("resolution", "720p")

        arguments: dict = {
            "prompt":       prompt,
            "image_url":    first_url,
            "aspect_ratio": kwargs.get("aspect_ratio", "16:9"),
            "duration":     duration_val,
        }

        if model_info.get("resolutions"):
            arguments["resolution"] = resolution_display

        if last_url:
            if "kling" in model_id:
                arguments["tail_image_url"] = last_url
            else:
                arguments["end_image_url"] = last_url

        seed = kwargs.get("seed", -1)
        if seed not in (-1, None):
            arguments["seed"] = int(seed)

        if model_info.get("has_audio") and kwargs.get("generate_audio") is not None:
            arguments["generate_audio"] = bool(kwargs["generate_audio"])

        if model_info.get("has_camera_fixed") and kwargs.get("camera_fixed") is not None:
            arguments["camera_fixed"] = bool(kwargs["camera_fixed"])

        if progress_cb:
            progress_cb("Submitting job…")

        def _on_update(update):
            if cancel_check and cancel_check():
                return
            if hasattr(update, "logs") and update.logs:
                msg = update.logs[-1].get("message", "") if isinstance(update.logs[-1], dict) else str(update.logs[-1])
                if progress_cb and msg:
                    progress_cb(msg[:80])
            elif progress_cb:
                progress_cb(type(update).__name__)

        try:
            result = fal_client.subscribe(
                model_id,
                arguments=arguments,
                with_logs=True,
                on_queue_update=_on_update,
            )
        except Exception as exc:
            if cancel_check and cancel_check():
                raise FalError("Cancelled by user.")
            raise FalError(str(exc)) from exc

        video_url = _extract_video_url(result)
        if not video_url:
            raise FalError(f"No video URL in result: {str(result)[:300]}")

        if progress_cb:
            progress_cb("Downloading…")
        return _download(video_url, dest_path)


# ═══════════════════════════════════════════════════════════════════════════════
#  Kling Direct client  (api.klingai.com)
# ═══════════════════════════════════════════════════════════════════════════════

class KlingClient:
    provider = "kling"
    BASE_URL = "https://api.klingai.com"

    def __init__(self, access_key: str, secret_key: str):
        self._ak = access_key.strip()
        self._sk = secret_key.strip()

    def _jwt_token(self) -> str:
        import jwt
        now = int(time.time())
        headers = {"alg": "HS256", "typ": "JWT"}
        payload = {
            "iss": self._ak,
            "exp": now + 1800,
            "nbf": now - 5,
        }
        return jwt.encode(payload, self._sk, algorithm="HS256", headers=headers)

    def _headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._jwt_token()}",
        }

    @staticmethod
    def _image_to_base64(path: str) -> str:
        p = Path(path)
        ext = p.suffix.lower()
        if ext in (".jpg", ".jpeg", ".png"):
            data = p.read_bytes()
        else:
            data = _to_jpeg_bytes(path)
        return base64.b64encode(data).decode("ascii")

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
        import requests

        model_info = next((m for m in MODELS.values()
                           if m["id"] == model_id
                           and m.get("kling_mode") == kwargs.get("kling_mode", "pro")),
                          next((m for m in MODELS.values() if m["id"] == model_id), {}))

        # ── encode frames ─────────────────────────────────────────────────────
        if progress_cb:
            progress_cb("Encoding first frame…")
        first_b64 = self._image_to_base64(first_frame)

        last_b64: Optional[str] = None
        if last_frame and model_info.get("has_end_frame"):
            if progress_cb:
                progress_cb("Encoding last frame…")
            last_b64 = self._image_to_base64(last_frame)

        try:
            duration_int = int(kwargs.get("duration", 5))
        except (TypeError, ValueError):
            duration_int = 5

        mode = kwargs.get("kling_mode", "pro")

        # ── build request body ────────────────────────────────────────────────
        body: dict = {
            "model_name": model_id,
            "mode":       mode,
            "duration":   str(duration_int),
            "image":      first_b64,
            "prompt":     prompt,
        }

        if last_b64:
            body["image_tail"] = last_b64

        cfg = kwargs.get("cfg_scale")
        if cfg is not None:
            body["cfg_scale"] = float(cfg)

        neg = kwargs.get("negative_prompt")
        if neg:
            body["negative_prompt"] = neg

        # ── submit task ───────────────────────────────────────────────────────
        if progress_cb:
            progress_cb("Submitting job…")

        try:
            resp = requests.post(
                f"{self.BASE_URL}/v1/videos/image2video",
                headers=self._headers(),
                json=body,
                timeout=30,
            )
            resp.raise_for_status()
        except Exception as exc:
            raise KlingError(f"Submit failed: {exc}") from exc

        rj = resp.json()
        if rj.get("code") != 0:
            raise KlingError(f"Kling API error: {rj.get('message', rj)}")

        task_id = rj.get("data", {}).get("task_id")
        if not task_id:
            raise KlingError(f"No task_id in response: {str(rj)[:300]}")

        # ── poll for completion ───────────────────────────────────────────────
        if progress_cb:
            progress_cb("Processing…")

        deadline = time.time() + MAX_WAIT
        while time.time() < deadline:
            if cancel_check and cancel_check():
                raise KlingError("Cancelled by user.")

            time.sleep(POLL_INTERVAL)

            try:
                poll = requests.get(
                    f"{self.BASE_URL}/v1/videos/image2video/{task_id}",
                    headers=self._headers(),
                    timeout=30,
                )
                poll.raise_for_status()
            except Exception as exc:
                if progress_cb:
                    progress_cb(f"Poll error: {exc}")
                continue

            pj = poll.json()
            data = pj.get("data", {})
            status = data.get("task_status", "")

            if status == "succeed":
                videos = data.get("task_result", {}).get("videos", [])
                if videos:
                    video_url = videos[0].get("url", "")
                    if video_url:
                        if progress_cb:
                            progress_cb("Downloading…")
                        return _download(video_url, dest_path)
                raise KlingError(f"Task succeeded but no video URL: {str(data)[:300]}")

            if status == "failed":
                msg = data.get("task_status_msg", "Unknown error")
                raise KlingError(f"Kling generation failed: {msg}")

            if progress_cb:
                progress_cb(f"Processing… ({status})")

        raise KlingError(f"Timeout after {MAX_WAIT}s waiting for task {task_id}")


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


def load_client(provider: str = "higgsfield"):
    """Load credentials from koan_config.json and return the appropriate client."""
    cfg_path = Path(__file__).parent / "koan_config.json"
    if not cfg_path.exists():
        raise HiggsfieldError(
            "koan_config.json not found. Add your API keys via API Keys."
        )
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    if provider == "fal":
        key = cfg.get("fal_api_key", "").strip()
        if not key:
            raise FalError(
                "fal_api_key not set. Open API Keys in the VIDEO tab."
            )
        return FalClient(key)

    if provider == "kling":
        ak = cfg.get("kling_access_key", "").strip()
        sk = cfg.get("kling_secret_key", "").strip()
        if not ak or not sk:
            raise KlingError(
                "kling_access_key / kling_secret_key not set. Open API Keys in the VIDEO tab."
            )
        return KlingClient(ak, sk)

    # default: higgsfield
    key    = cfg.get("higgsfield_api_key",    "").strip()
    secret = cfg.get("higgsfield_api_secret", "").strip()
    if not key:
        raise HiggsfieldError(
            "higgsfield_api_key not set. Open API Keys in the VIDEO tab."
        )
    return HiggsfieldClient(key, secret)
