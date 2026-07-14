"""Runtime composition for the root ``potpie`` distribution."""

from potpie.runtime.composition import (
    LocalEngineClient,
    PotpieRuntime,
    create_runtime,
    get_runtime,
    reset_runtime,
)
from potpie.runtime.settings import ProductSettings
from potpie.skills.resource_provider import (
    ROOT_TEMPLATE_RESOURCES,
    TemplateResourceProvider,
)


def cli_template_resources() -> TemplateResourceProvider:
    return ROOT_TEMPLATE_RESOURCES


__all__ = [
    "LocalEngineClient",
    "PotpieRuntime",
    "ProductSettings",
    "cli_template_resources",
    "create_runtime",
    "get_runtime",
    "reset_runtime",
]
