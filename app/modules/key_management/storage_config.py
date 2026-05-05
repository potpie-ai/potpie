import os
from typing import Optional


TRUE_VALUES = {"true", "1", "yes", "on", "enabled"}
FALSE_VALUES = {"false", "0", "no", "off", "disabled"}


def parse_env_bool(value: Optional[str]) -> Optional[bool]:
    """Parse common boolean env values, returning None for unset/unknown values."""
    if value is None:
        return None

    normalized = value.strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    return None


def is_gcp_secret_manager_enabled() -> bool:
    """Return whether secret storage should attempt Google Secret Manager.

    Supports the current positive flag and the legacy negative flag. When the
    two conflict, prefer disabling GCP so local/dev environments do not make
    accidental cloud calls.
    """
    enabled = parse_env_bool(os.environ.get("GCP_SECRET_MANAGER_ENABLED"))
    disabled = parse_env_bool(os.environ.get("GCP_SECRET_MANAGER_DISABLED"))

    if enabled is False or disabled is True:
        return False
    if enabled is True:
        return True
    return True
