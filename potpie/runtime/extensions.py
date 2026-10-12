"""Extension operations an embedding host registers on its endpoint.

An embedding host can serve operations of its own product beside the engine
and host catalogs, on the same typed envelope, without Potpie's catalog
carrying them. Each is a :class:`HostExtension` named ``ext.<vendor>.<op>``;
Potpie defines none. Extensions follow the same evolution rules as every
other operation (compact requests, precise refusals, lenient results), but
their schemas belong to the host that registers them.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, is_dataclass
from typing import Final

from potpie.runtime.host_operations import HOST_OPERATION_CATALOG, HostOperation
from potpie.runtime.operations import (
    ENGINE_OPERATION_CATALOG,
    EngineOperation,
    OperationSpec,
    SafetyClass,
)
from potpie.runtime.resource_manager import AuthenticatedActor
from potpie_context_engine import ContextIdentity

#: ``ext.<vendor>.<op>``: lower-case identifiers, one vendor segment.
EXTENSION_NAME: Final = re.compile(r"^ext\.[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$")

ExtensionHandler = Callable[..., Awaitable[object]]
"""``async handler(request, *, actor, context)`` returning the result type or
an outcome; ``context`` is ``None`` unless the extension requires one."""


@dataclass(frozen=True, slots=True)
class HostExtension:
    """One extension operation and the handler that serves it."""

    name: str
    request_type: type
    result_type: type
    safety: SafetyClass
    handler: ExtensionHandler
    context_required: bool = False
    destructive: bool = False
    resource_type: str | None = None
    resource_identity_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not EXTENSION_NAME.fullmatch(self.name):
            raise ValueError(
                f"extension operation names must look like 'ext.<vendor>.<op>', "
                f"not {self.name!r}"
            )
        for kind, value in (
            ("request", self.request_type),
            ("result", self.result_type),
        ):
            if not (isinstance(value, type) and is_dataclass(value)):
                raise TypeError(f"{self.name}: the {kind} type must be a dataclass")
        if self.safety is SafetyClass.DAEMON_LIFECYCLE_CONTROL:
            raise ValueError(f"{self.name}: extensions cannot control the host")
        self.spec  # noqa: B018 - validates the resource metadata eagerly

    @property
    def spec(self) -> OperationSpec:
        return OperationSpec(
            self.name,
            self.request_type,
            self.result_type,
            self.safety,
            destructive=self.destructive,
            context_required=self.context_required,
            resource_type=self.resource_type,
            resource_identity_fields=self.resource_identity_fields,
        )

    async def handle(
        self,
        request: object,
        *,
        actor: AuthenticatedActor,
        context: ContextIdentity | None,
    ) -> object:
        return await self.handler(request, actor=actor, context=context)


def extension_registry(
    extensions: Iterable[HostExtension],
) -> dict[str, HostExtension]:
    """Index extensions by name, refusing duplicates."""

    registry: dict[str, HostExtension] = {}
    for extension in extensions:
        if extension.name in registry:
            raise ValueError(f"extension {extension.name!r} is registered twice")
        registry[extension.name] = extension
    return registry


def operation_spec(
    name: str, *, extensions: Iterable[HostExtension] = ()
) -> OperationSpec | None:
    """The spec of an engine, host or given extension operation, by wire name.

    The one lookup an embedding host's authorizer needs to decide by safety
    class and destructiveness. ``None`` for a name nothing serves.
    """

    try:
        return ENGINE_OPERATION_CATALOG[EngineOperation(name)]
    except ValueError:
        pass
    try:
        return HOST_OPERATION_CATALOG[HostOperation(name)]
    except ValueError:
        pass
    for extension in extensions:
        if extension.name == name:
            return extension.spec
    return None


__all__ = [
    "EXTENSION_NAME",
    "ExtensionHandler",
    "HostExtension",
    "extension_registry",
    "operation_spec",
]
