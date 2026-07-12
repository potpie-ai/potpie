"""Template resource providers for packaged Potpie agent skill bundles."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from importlib.resources.abc import Traversable
from typing import Protocol


class TemplateResourceProvider(Protocol):
    """Provides the package/file root that owns bundled agent templates."""

    def files_root(self) -> Traversable:
        """Return a traversable root containing the ``templates/`` directory."""


@dataclass(frozen=True, slots=True)
class PackageTemplateResources:
    """Read bundled templates from an importable package."""

    package: str

    def files_root(self) -> Traversable:
        return resources.files(self.package)


@dataclass(frozen=True, slots=True)
class MissingTemplateResources:
    """Explicit no-resource provider for library-only context-engine usage."""

    capability: str = "skills.template_resources"

    def files_root(self) -> Traversable:
        raise RuntimeError(
            "No packaged Potpie skill resources were provided; run this "
            "operation through the root 'potpie' distribution."
        )


NO_TEMPLATE_RESOURCES = MissingTemplateResources()
ROOT_TEMPLATE_RESOURCES = PackageTemplateResources("potpie.skills.resources")


def resolve_template_resources(
    provider: TemplateResourceProvider | None = None,
) -> TemplateResourceProvider:
    """Normalize an optional provider to an explicit no-resource provider."""

    return provider if provider is not None else NO_TEMPLATE_RESOURCES


__all__ = [
    "MissingTemplateResources",
    "NO_TEMPLATE_RESOURCES",
    "PackageTemplateResources",
    "ROOT_TEMPLATE_RESOURCES",
    "TemplateResourceProvider",
    "resolve_template_resources",
]
