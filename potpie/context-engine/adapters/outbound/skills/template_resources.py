"""Template resource providers for packaged Potpie agent skill bundles."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from importlib.resources.abc import Traversable
from typing import Protocol

from domain.errors import CapabilityNotImplemented


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
        raise CapabilityNotImplemented(
            self.capability,
            detail="No packaged Potpie CLI template resources were provided.",
            recommended_next_action=(
                "run this command through the root 'potpie' CLI or pass a "
                "TemplateResourceProvider into build_host_shell"
            ),
        )


NO_TEMPLATE_RESOURCES = MissingTemplateResources()


def resolve_template_resources(
    provider: TemplateResourceProvider | None = None,
) -> TemplateResourceProvider:
    """Normalize an optional provider to an explicit no-resource provider."""

    return provider if provider is not None else NO_TEMPLATE_RESOURCES


__all__ = [
    "MissingTemplateResources",
    "NO_TEMPLATE_RESOURCES",
    "PackageTemplateResources",
    "TemplateResourceProvider",
    "resolve_template_resources",
]
