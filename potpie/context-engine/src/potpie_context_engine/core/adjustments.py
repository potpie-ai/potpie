"""Disclosed request adjustments: the ``requested`` / ``effective`` contract.

A read that ran with a bounded or canonicalized version of what was asked for
is still a successful read — *provided the change is disclosed alongside the
data*. ``--depth 100`` on a view whose backend walks at most four hops should
return the depth-4 neighbourhood with one notice, not a refusal that costs the
caller a retry, and ``--type repository`` should return the ``Repository`` rows
rather than a confident empty answer.

This module is the one shape every surface uses to say so. The CLI attaches
the list to its envelopes (``status="adjusted"`` + ``adjustments``), the text
renderer prints one line per entry, and services that normalize on their own
(the RPC path) can carry the same records back as warnings. Keeping the
vocabulary of reason codes small is deliberate: an agent branches on the code,
a human reads the message.

Boundaries the contract does *not* cross: an adjustment never changes the
selected pot, entity, environment, or revision, never widens a write, and
never guesses at a near-miss token. Those remain refusals with candidates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Iterable, Mapping

#: ``status`` value an envelope carries when at least one adjustment applied.
STATUS_ADJUSTED: Final[str] = "adjusted"

#: Reason codes. Additive: a consumer that does not know a code treats the
#: entry as an informational notice.
REASON_MAXIMUM_SUPPORTED: Final[str] = "maximum_supported"
REASON_CANONICAL_ALIAS: Final[str] = "canonical_alias"
REASON_CANONICAL_CASE: Final[str] = "canonical_case"
REASON_EXPLICIT_SINCE: Final[str] = "explicit_since"
REASON_UNIT_ALIAS: Final[str] = "unit_alias"
REASON_MACHINE_JSON: Final[str] = "machine_json"
REASON_DEFAULT_APPLIED: Final[str] = "default_applied"

#: The one-character marker text output uses for an adjustment line, kept
#: distinct from ``!`` (warnings) so a reader can tell "handled, disclosed"
#: from "something to act on".
TEXT_MARKER: Final[str] = "~"


@dataclass(frozen=True, slots=True)
class Adjustment:
    """One disclosed change between what was requested and what ran."""

    field: str
    requested: Any
    effective: Any
    reason: str
    message: str
    max_supported: Any = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "field": self.field,
            "requested": self.requested,
            "effective": self.effective,
            "reason": self.reason,
            "message": self.message,
        }
        if self.max_supported is not None:
            out["max_supported"] = self.max_supported
        return out

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (self.field, str(self.requested), str(self.effective), self.reason)


def dedupe_adjustments(items: Iterable[Adjustment]) -> tuple[Adjustment, ...]:
    """Stable order, one entry per (field, requested, effective, reason).

    Two code paths may legitimately observe the same adjustment — the CLI
    normalizing before dispatch and a presenter normalizing again when it
    renders — and the caller must see the notice once.
    """
    seen: set[tuple[str, str, str, str]] = set()
    out: list[Adjustment] = []
    for item in items:
        if item.key in seen:
            continue
        seen.add(item.key)
        out.append(item)
    return tuple(out)


def adjustment_dicts(items: Iterable[Adjustment]) -> list[dict[str, Any]]:
    return [item.to_dict() for item in dedupe_adjustments(items)]


def adjustment_lines(items: Iterable[Adjustment]) -> list[str]:
    """The text-mode rendering: one marked line per adjustment."""
    return [f"{TEXT_MARKER} {item.message}" for item in dedupe_adjustments(items)]


def with_adjustments(
    payload: Mapping[str, Any], items: Iterable[Adjustment]
) -> dict[str, Any]:
    """``payload`` plus ``status``/``adjustments`` when anything was adjusted.

    Additive on purpose: a payload with no adjustments is returned byte-for-
    byte unchanged so existing JSON consumers see exactly what they saw before.
    A payload that already reports a non-``ok`` status keeps it — a refused or
    partial result is never relabelled as merely adjusted.
    """
    deduped = dedupe_adjustments(items)
    out = dict(payload)
    if not deduped:
        return out
    out["adjustments"] = [item.to_dict() for item in deduped]
    current = out.get("status")
    if current in (None, "", "ok"):
        out["status"] = STATUS_ADJUSTED
    return out


def adjustments_from_dicts(value: Any) -> tuple[Adjustment, ...]:
    """Rehydrate adjustments carried inside a serialized payload."""
    if not isinstance(value, (list, tuple)):
        return ()
    out: list[Adjustment] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        try:
            out.append(
                Adjustment(
                    field=str(item["field"]),
                    requested=item.get("requested"),
                    effective=item.get("effective"),
                    reason=str(item.get("reason") or "adjusted"),
                    message=str(item.get("message") or ""),
                    max_supported=item.get("max_supported"),
                )
            )
        except KeyError:
            continue
    return dedupe_adjustments(out)


__all__ = [
    "Adjustment",
    "REASON_CANONICAL_ALIAS",
    "REASON_CANONICAL_CASE",
    "REASON_DEFAULT_APPLIED",
    "REASON_EXPLICIT_SINCE",
    "REASON_MACHINE_JSON",
    "REASON_MAXIMUM_SUPPORTED",
    "REASON_UNIT_ALIAS",
    "STATUS_ADJUSTED",
    "TEXT_MARKER",
    "adjustment_dicts",
    "adjustment_lines",
    "adjustments_from_dicts",
    "dedupe_adjustments",
    "with_adjustments",
]
