from __future__ import annotations

import uuid

# Stable namespace for deterministic seed UIDs (UUID namespace URL)
_SEED_NAMESPACE = uuid.UUID("6ba7b811-9dad-11d1-80b4-00c04fd430c8")


def deterministic_seed_uid(profile: str, email: str) -> str:
    """Return a stable user uid string for a profile + email (UUID v5)."""
    key = f"potpie-seed:{profile}:{email.strip().lower()}"
    return str(uuid.uuid5(_SEED_NAMESPACE, key))
