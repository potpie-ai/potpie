"""What a host says it serves, asked once per connection.

The RPC allowlist is maintained twice — here and in the managed service's own
build — and the copies drifted without anything noticing. ``surfaces.py`` is
this repo's half of the answer: it declares the set and publishes it at
``GET /surfaces``. This module is the *client's* half: it asks a host that
question, once per endpoint-and-credential, and remembers the answer for the
life of the process.

Why that matters more than it sounds: before this, the only way the CLI could
tell "this host does not implement that surface" from "you called it wrong" was
to read the other service's prose — ``message.startswith("invalid RPC surface:
")``. That is a contract nobody agreed to. The managed team can reword one
string and silently turn a capability gap back into ``validation_error``/exit 1
("the caller got it wrong", no repair offered) for a caller that got it right,
which is exactly the failure that killed ``potpie doctor`` against managed. With
a negotiated set the decision is structural: the host itself said which surfaces
it serves, so a refusal on one it did not list *is* a capability gap whatever
words it arrives in — and, just as importantly, a refusal on one it *did* list
is never retargeted, however the message happens to read.

Three rules hold this together:

* **Silence is not emptiness.** A host that answers 404, refuses the credential,
  or cannot be reached has an *unknown* surface list, never an empty one. Any
  other reading would have this client refuse commands that work perfectly well
  against every deployment older than the endpoint. :data:`UNKNOWN` is that
  answer, and every failure collapses to it.
* **Never stricter than the host.** Nothing here gates a call. The negotiated
  set only classifies a refusal the host has already sent, so a host that
  publishes a stale or partial list still gets to answer for itself.
* **One round trip.** The answer is cached per ``(base_url, token)`` — the
  credential is part of the key because a 401 is one of the outcomes recorded as
  "does not say", and a key rotated inside one long-lived process must not keep
  reading the old refusal.

The ask is made the first time an answer would change what the caller is told,
not on connect: a host that publishes ``/surfaces`` is by construction a host
that already answers an unserved surface with ``not_implemented``
(:func:`potpie.daemon.main._validate_rpc_target`), so probing eagerly would add
a request to every healthy session to learn something no healthy session uses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

#: Seconds to wait for a host to describe itself. Short and fixed: this runs on
#: a path that is already failing, and a host slow to answer a question about
#: *itself* is one more thing not worth waiting on before reporting the refusal
#: the caller is standing there holding.
NEGOTIATION_TIMEOUT_S: Final[float] = 5.0


@dataclass(frozen=True, slots=True)
class HostContract:
    """One host's answer to "what do you serve?", or its silence.

    ``surfaces is None`` means the host did not say. It never means the host
    serves nothing.
    """

    surfaces: frozenset[str] | None = None
    contract: int | None = None

    @property
    def answered(self) -> bool:
        return self.surfaces is not None

    def serves(self, surface: str) -> bool | None:
        """``True``/``False`` when the host said, ``None`` when it did not."""
        if self.surfaces is None:
            return None
        return surface in self.surfaces


#: The answer for every host that did not give one.
UNKNOWN: Final[HostContract] = HostContract()

#: One entry per ``(base_url, token)``; see the module docstring on the key.
_negotiated: dict[tuple[str, str], HostContract] = {}


def negotiate(base_url: str, token: str) -> HostContract:
    """What ``base_url`` says it serves, asked at most once per credential."""
    key = (base_url.rstrip("/"), token)
    cached = _negotiated.get(key)
    if cached is not None:
        return cached
    answer = _ask(*key)
    _negotiated[key] = answer
    return answer


def serves(base_url: str, token: str, surface: str) -> bool | None:
    """Does this host serve ``surface``? ``None`` when it does not say."""
    return negotiate(base_url, token).serves(surface)


def _ask(base_url: str, token: str) -> HostContract:
    """One authenticated GET, with every failure collapsed to "does not say".

    Every failure: a 404 from a host that predates the endpoint, a rejected
    credential, a proxy's HTML, an unreachable address, a body in a shape this
    client does not understand. Raising here would mean a diagnostic aborting
    the report of the error it was called to explain.
    """
    if not base_url.startswith(("http://", "https://")):
        return UNKNOWN

    import httpx

    try:
        response = httpx.get(
            f"{base_url}/surfaces",
            headers={"Authorization": f"Bearer {token}"},
            timeout=NEGOTIATION_TIMEOUT_S,
        )
        if response.status_code != 200:
            return UNKNOWN
        body = response.json()
    except Exception:  # noqa: BLE001 - transport, JSON, anything: it did not say
        return UNKNOWN
    return _read(body)


def _read(body: Any) -> HostContract:
    """The published contract, or ``UNKNOWN`` for anything unrecognisable.

    A list this client cannot read as surface names is not an empty set — it is
    a host answering in a dialect this build does not know, which is what
    ``contract`` exists to make visible if the shape ever changes.
    """
    if not isinstance(body, dict):
        return UNKNOWN
    published = body.get("surfaces")
    if not isinstance(published, list):
        return UNKNOWN
    if any(not isinstance(name, str) for name in published):
        return UNKNOWN
    version = body.get("contract")
    return HostContract(
        surfaces=frozenset(published),
        contract=version if isinstance(version, int) else None,
    )


def reset() -> None:
    """Forget every negotiated answer. Tests, and registry writes."""
    _negotiated.clear()


__all__ = [
    "NEGOTIATION_TIMEOUT_S",
    "UNKNOWN",
    "HostContract",
    "negotiate",
    "reset",
    "serves",
]
