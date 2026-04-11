"""KOAN.img auto-updater — checks GitHub Releases for new versions."""
from __future__ import annotations

import io
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Optional, Tuple
from urllib import request, error

from version import __version__

REPO_OWNER = "ahamitchell"
REPO_NAME = "KOAN-IMG"
API_URL = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/releases/latest"

# Files/dirs that belong to the user, never overwritten by updates
USER_FILES = {
    "koan_config.json",
    ".koan_ui_state.json",
    "video_state.json",
    "video_edit_library.json",
    "narrative_state.json",
}


def _parse_version(v: str) -> Tuple[int, ...]:
    """Parse 'v1.2.3' or '1.2.3' into (1, 2, 3)."""
    v = v.lstrip("vV").strip()
    parts = []
    for p in v.split("."):
        try:
            parts.append(int(p))
        except ValueError:
            break
    return tuple(parts) or (0,)


def check_for_update() -> Optional[dict]:
    """Check GitHub for a newer release.

    Returns dict with 'tag', 'name', 'body', 'zip_url' if update available,
    or None if already up to date (or on error).
    """
    try:
        req = request.Request(API_URL, headers={"Accept": "application/vnd.github+json"})
        with request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
    except (error.URLError, json.JSONDecodeError, OSError):
        return None

    remote_tag = data.get("tag_name", "")
    remote_ver = _parse_version(remote_tag)
    local_ver = _parse_version(__version__)

    if remote_ver <= local_ver:
        return None

    return {
        "tag": remote_tag,
        "name": data.get("name", remote_tag),
        "body": data.get("body", ""),
        "zip_url": data.get("zipball_url", ""),
    }


def apply_update(zip_url: str, app_dir: Optional[Path] = None) -> bool:
    """Download the release zip and replace app files (preserving user data).

    Returns True on success, False on failure.
    """
    if app_dir is None:
        app_dir = Path(__file__).resolve().parent

    try:
        req = request.Request(zip_url, headers={"Accept": "application/vnd.github+json"})
        with request.urlopen(req, timeout=120) as resp:
            zip_bytes = resp.read()
    except (error.URLError, OSError):
        return False

    try:
        tmp_dir = Path(tempfile.mkdtemp(prefix="koan_update_"))

        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            zf.extractall(tmp_dir)

        # GitHub zipball has a top-level folder like "owner-repo-hash/"
        extracted = [d for d in tmp_dir.iterdir() if d.is_dir()]
        if not extracted:
            return False
        repo_root = extracted[0]

        # The repo has code inside ai_photo_picker/ — that maps to {app}/ on disk
        src_dir = repo_root / "ai_photo_picker"
        if not src_dir.is_dir():
            # Fallback: maybe the zip structure changed
            src_dir = repo_root

        # Copy new files over old ones, skipping user data
        for item in src_dir.rglob("*"):
            if item.is_dir():
                continue
            rel = item.relative_to(src_dir)
            if rel.name in USER_FILES:
                continue
            dest = app_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, dest)

        # Update local version file
        version_file = app_dir / "version.py"
        # The new version.py from the zip already has the correct version
        # so no extra step needed here

        return True

    except Exception:
        return False

    finally:
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass
