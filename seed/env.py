from __future__ import annotations

from dotenv import load_dotenv

from seed.paths import repo_root


def load_seed_env() -> None:
    """Load the same `.env` as the app (repo root), overriding prior values."""
    load_dotenv(repo_root() / ".env", override=True)
