"""KOAN.img Launcher — bootstraps Python environment and launches the app.

This script is compiled to a small .exe via PyInstaller.
It handles first-run setup (venv, pip install) and update checks.
"""
from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

# ── paths ────────────────────────────────────────────────────────────────────
LAUNCHER_DIR = Path(__file__).resolve().parent
APP_DIR = LAUNCHER_DIR / "app"
VENV_DIR = LAUNCHER_DIR / "venv"
PYTHON_DIR = LAUNCHER_DIR / "python"
REQUIREMENTS = APP_DIR / "requirements.txt"

if platform.system() == "Windows":
    VENV_PYTHON = VENV_DIR / "Scripts" / "python.exe"
    VENV_PIP = VENV_DIR / "Scripts" / "pip.exe"
else:
    VENV_PYTHON = VENV_DIR / "bin" / "python"
    VENV_PIP = VENV_DIR / "bin" / "pip"

# If we bundled embedded Python, use that; otherwise use system Python
SYSTEM_PYTHON = PYTHON_DIR / "python.exe" if (PYTHON_DIR / "python.exe").exists() else Path(sys.executable)


def _print_status(msg: str) -> None:
    """Print a status message (visible in console window)."""
    print(f"[KOAN.img] {msg}")


def _run(cmd: list, check: bool = True, **kwargs) -> subprocess.CompletedProcess:
    """Run a subprocess with visible output."""
    return subprocess.run(cmd, check=check, **kwargs)


def ensure_venv() -> None:
    """Create virtual environment if it doesn't exist."""
    if VENV_PYTHON.exists():
        return

    _print_status("Creating virtual environment...")
    # Embedded Python doesn't have venv module, use virtualenv instead
    _run([str(SYSTEM_PYTHON), "-m", "virtualenv", str(VENV_DIR)],
         check=False)
    if not VENV_PYTHON.exists():
        # Fallback to venv if virtualenv not available (system Python)
        _run([str(SYSTEM_PYTHON), "-m", "venv", str(VENV_DIR)])

    if not VENV_PYTHON.exists():
        _print_status("ERROR: Failed to create virtual environment.")
        _print_status(f"Tried using Python at: {SYSTEM_PYTHON}")
        input("Press Enter to exit...")
        sys.exit(1)

    _print_status("Virtual environment created.")


def ensure_dependencies() -> None:
    """Install/update pip dependencies."""
    if not REQUIREMENTS.exists():
        _print_status("WARNING: requirements.txt not found, skipping dependency install.")
        return

    marker = VENV_DIR / ".deps_installed"

    # Check if requirements changed since last install
    import hashlib
    req_hash = hashlib.md5(REQUIREMENTS.read_bytes()).hexdigest()

    if marker.exists() and marker.read_text().strip() == req_hash:
        return  # Already up to date

    _print_status("Installing dependencies (this may take a few minutes on first run)...")

    # Upgrade pip first
    _run([str(VENV_PYTHON), "-m", "pip", "install", "--upgrade", "pip"],
         check=False)

    # Install PyTorch with CUDA support if available
    _print_status("Installing PyTorch...")
    _run([str(VENV_PIP), "install",
          "torch", "torchvision",
          "--index-url", "https://download.pytorch.org/whl/cu121"],
         check=False)

    # Install remaining dependencies
    _print_status("Installing remaining packages...")
    _run([str(VENV_PIP), "install", "-r", str(REQUIREMENTS)])

    marker.write_text(req_hash)
    _print_status("Dependencies installed.")


def check_update() -> None:
    """Check for updates via the updater module."""
    try:
        # Run the update check using the venv Python so imports work
        result = subprocess.run(
            [str(VENV_PYTHON), "-c",
             "from updater import check_for_update; "
             "info = check_for_update(); "
             "print(info if info else '')"],
            capture_output=True, text=True, timeout=15,
            cwd=str(APP_DIR),
        )
        output = result.stdout.strip()
        if output and output != "None":
            import ast
            info = ast.literal_eval(output)
            if isinstance(info, dict):
                tag = info.get("tag", "?")
                name = info.get("name", "")
                body = info.get("body", "")
                _print_status(f"")
                _print_status(f"  Update available: {name} ({tag})")
                if body:
                    # Show first 3 lines of changelog
                    for line in body.strip().splitlines()[:3]:
                        _print_status(f"    {line}")
                _print_status(f"")

                answer = input("  Install update now? [y/N]: ").strip().lower()
                if answer in ("y", "yes"):
                    _print_status("Downloading update...")
                    update_result = subprocess.run(
                        [str(VENV_PYTHON), "-c",
                         f"from updater import apply_update; "
                         f"ok = apply_update({info['zip_url']!r}); "
                         f"print('OK' if ok else 'FAIL')"],
                        capture_output=True, text=True, timeout=120,
                        cwd=str(APP_DIR),
                    )
                    if "OK" in update_result.stdout:
                        _print_status("Update installed! Restarting...")
                        # Re-check dependencies in case requirements changed
                        ensure_dependencies()
                    else:
                        _print_status("Update failed. Continuing with current version.")
    except Exception:
        pass  # Don't block app launch on update check failure


def launch_app() -> None:
    """Launch the main KOAN.img application."""
    app_entry = APP_DIR / "ui_app.py"

    if not app_entry.exists():
        _print_status(f"ERROR: {app_entry} not found.")
        input("Press Enter to exit...")
        sys.exit(1)

    _print_status("Launching KOAN.img...")
    proc = subprocess.run([str(VENV_PYTHON), str(app_entry)], cwd=str(APP_DIR))
    sys.exit(proc.returncode)


def main() -> None:
    _print_status("Starting up...")

    ensure_venv()
    ensure_dependencies()
    check_update()
    launch_app()


if __name__ == "__main__":
    main()
