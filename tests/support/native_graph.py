"""Own embedded graph clients acquired by one root test.

The engine's process registry is the native resource owner. Close its test-only
leases before pytest removes graph storage; keep preexisting clients and other
processes' live connections intact. Driver atexit callbacks remain registered.
"""

from __future__ import annotations

import sys
from typing import Any

_MODULE = "potpie_context_engine.adapters.outbound.graph.falkordb_writer"


class EmbeddedServerLease:
    def __init__(self) -> None:
        module = sys.modules.get(_MODULE)
        registry = getattr(module, "_OPEN_SERVERS", {})
        self._existing = {path: tuple(clients) for path, clients in registry.items()}

    def close(self) -> int:
        module = sys.modules.get(_MODULE)
        if module is None:
            return 0
        registry: dict[str, list[Any]] = module._OPEN_SERVERS
        stopped = 0
        errors = []
        for path, clients in list(registry.items()):
            previous = self._existing.get(path, ())
            owned = [
                client
                for client in clients
                if not any(client is old for old in previous)
            ]
            if not owned:
                continue
            retained = [
                client for client in clients if any(client is old for old in previous)
            ]
            try:
                if retained:
                    # A longer-lived fixture owns this server. Release only
                    # the current test's additional connection pools.
                    for client in owned:
                        module._without_retries(client)
                        client.connection_pool.disconnect()
                elif module._stop_if_unused(path, owned):
                    # The normal owner cleanup stops only an unused server;
                    # another process's active connection prevents shutdown.
                    stopped += 1
            except Exception as error:
                error.add_note(f"embedded graph test cleanup: {path}")
                errors.append(error)
                continue
            if retained:
                registry[path] = retained
            else:
                registry.pop(path, None)
        if errors:
            raise ExceptionGroup("Embedded graph test cleanup failed", errors)
        return stopped
