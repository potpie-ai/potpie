"""Per-pot ingestion configuration (port).

Two ingestion modes:

- ``immediate``: every admitted event enqueues its batch right away.
  Historical default — kept available for low-volume / latency-sensitive
  pots that want every event reflected in the graph ASAP.

- ``windowed``: admitted events accumulate in the pot's open batch but
  the batch is *not* enqueued. A scheduled task closes-and-enqueues the
  batch once it's older than ``window_minutes`` (or hits
  ``min_batch_size`` if set). The user can also force-flush at any time
  via the ``/ingest/flush`` endpoint.

The migration sets every existing pot to ``windowed`` with a 5-minute
window — see :func:`flush_ready_windowed_pots` for the dispatcher.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

IngestionMode = Literal["immediate", "windowed"]

_INGESTION_MODE_BY_VALUE: dict[str, IngestionMode] = {
    "immediate": "immediate",
    "windowed": "windowed",
}


def parse_ingestion_mode(value: str) -> IngestionMode:
    """Parse a stored/configured mode into the domain Literal."""
    try:
        return _INGESTION_MODE_BY_VALUE[value.strip().lower()]
    except KeyError as exc:
        raise ValueError(f"unknown ingestion mode: {value}") from exc


@dataclass(slots=True, frozen=True)
class IngestionConfig:
    """Effective ingestion config for one pot.

    The dataclass is also used as the in-memory default — code paths that
    can't reach the DB (tests, container build before DB is up) fall back
    to ``IngestionConfig.default()`` which is windowed/5min.
    """

    pot_id: str
    mode: IngestionMode
    window_minutes: int
    min_batch_size: int | None

    @classmethod
    def default(cls, pot_id: str) -> IngestionConfig:
        return cls(
            pot_id=pot_id,
            mode="windowed",
            window_minutes=5,
            min_batch_size=None,
        )


class IngestionConfigPort(Protocol):
    """Read + write the per-pot ingestion config."""

    def get(self, pot_id: str) -> IngestionConfig:
        """Return the config for ``pot_id``.

        Adapters must materialize the default (``windowed/5min``) for pots
        with no row so callers never have to special-case "no config".
        """
        ...

    def set(
        self,
        *,
        pot_id: str,
        mode: IngestionMode,
        window_minutes: int,
        min_batch_size: int | None,
        actor_user_id: str | None = None,
    ) -> IngestionConfig:
        """Upsert the config for ``pot_id`` and return the new effective value."""
        ...

    def list_windowed_pots_ready_to_flush(
        self, *, as_of_unix_seconds: float
    ) -> list[IngestionConfig]:
        """Return pot configs that are due for windowed flush.

        "Due" = ``mode='windowed'`` and either (a) the pot's open batch is
        older than ``window_minutes``, or (b) ``min_batch_size`` is set and
        the open batch has reached it. The caller decides whether to actually
        close + enqueue — the port only surfaces the candidates.
        """
        ...


class InMemoryIngestionConfig:
    """In-memory adapter for tests. Defaults match the production default."""

    def __init__(
        self,
        configs: dict[str, IngestionConfig] | None = None,
        *,
        pending_batches_by_pot: dict[str, list[float]] | None = None,
    ) -> None:
        self._configs = dict(configs or {})
        # Stored as: pot_id → list of open-batch creation timestamps (seconds).
        # Tests configure these to simulate stale pending batches.
        self._pending_batches = dict(pending_batches_by_pot or {})

    def get(self, pot_id: str) -> IngestionConfig:
        return self._configs.get(pot_id) or IngestionConfig.default(pot_id)

    def set(
        self,
        *,
        pot_id: str,
        mode: IngestionMode,
        window_minutes: int,
        min_batch_size: int | None,
        actor_user_id: str | None = None,
    ) -> IngestionConfig:
        del actor_user_id  # Recorded by the SQL adapter; test stub ignores it.
        cfg = IngestionConfig(
            pot_id=pot_id,
            mode=mode,
            window_minutes=window_minutes,
            min_batch_size=min_batch_size,
        )
        self._configs[pot_id] = cfg
        return cfg

    def list_windowed_pots_ready_to_flush(
        self, *, as_of_unix_seconds: float
    ) -> list[IngestionConfig]:
        out: list[IngestionConfig] = []
        for pot_id, cfg in self._configs.items():
            if cfg.mode != "windowed":
                continue
            batches = self._pending_batches.get(pot_id, [])
            for created_at in batches:
                if as_of_unix_seconds - created_at >= cfg.window_minutes * 60:
                    out.append(cfg)
                    break
        return out


__all__ = [
    "IngestionConfig",
    "IngestionConfigPort",
    "IngestionMode",
    "InMemoryIngestionConfig",
    "parse_ingestion_mode",
]
