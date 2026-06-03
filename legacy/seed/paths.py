from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """Directory containing pyproject.toml (Potpie repo root)."""
    p = Path(__file__).resolve().parent
    for _ in range(8):
        if (p / "pyproject.toml").exists():
            return p
        if p == p.parent:
            break
        p = p.parent
    raise RuntimeError(
        "Could not locate repo root (no pyproject.toml in parents of seed package)"
    )


def seed_dir() -> Path:
    return repo_root() / "seed"


def data_dir(profile: str) -> Path:
    return seed_dir() / "data" / profile


def state_path(profile: str) -> Path:
    return seed_dir() / ".state" / f"{profile}.json"
