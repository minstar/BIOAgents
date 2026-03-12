"""BIOAgents - Medical/Biomedical Agent GYM Framework."""

__version__ = "0.1.0"

import sys
from pathlib import Path

# ── Canonical project root & venv Python ──────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

_VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
PYTHON_EXE: str = str(_VENV_PYTHON) if _VENV_PYTHON.exists() else sys.executable
"""Path to the project-specific Python interpreter.

IMPORTANT — Always use this when spawning subprocesses:

    import subprocess
    from bioagents import PYTHON_EXE
    subprocess.run([PYTHON_EXE, "script.py"])

The project uses a dedicated virtual environment at `.venv/` (created with `uv`).
Using conda base or system Python will cause missing-module errors (bioagents,
vllm, flash-attn, etc.).
"""
