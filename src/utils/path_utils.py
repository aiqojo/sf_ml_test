"""Path helpers for project utilities."""

from pathlib import Path
from typing import Iterable, Optional


DEFAULT_MARKERS = (
    "pixi.toml",
    "requirements.txt",
    "pyproject.toml",
    "README.md",
    ".git",
)


def get_repo_root(
    start_path: Optional[str] = None,
    markers: Optional[Iterable[str]] = None,
) -> Path:
    """Find repository root by walking up from start_path.

    Defaults to this file's location and common repo markers.
    """
    markers = tuple(markers) if markers is not None else DEFAULT_MARKERS
    start = Path(start_path).resolve() if start_path else Path(__file__).resolve()
    start_dir = start if start.is_dir() else start.parent

    for path in (start_dir, *start_dir.parents):
        for marker in markers:
            if (path / marker).exists():
                return path

    return start_dir
