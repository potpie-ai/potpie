"""Root-owned product configuration."""

from potpie.config.service import (
    KNOWN_CONFIG_KEYS,
    ProductConfigService,
    is_secret_config_key,
    public_config_value,
)

__all__ = [
    "KNOWN_CONFIG_KEYS",
    "ProductConfigService",
    "is_secret_config_key",
    "public_config_value",
]
