"""The served RPC surface set is pinned, and compared with the other copy of it.

The allowlist guarding ``POST /rpc`` exists in more than one place and has never
had a negotiation protocol. It exists here as
:data:`potpie.daemon.surfaces.RPC_SURFACES`, and again in the managed service's
own build (``github.com/potpie-ai/pie``), which cannot import this package —
that repo depends on ``potpie-context-engine``, not on the ``potpie`` product
distribution, so its copy is transcribed by hand.

They drifted, and the drift was silent until a user hit it: the managed copy had
no ``resources`` entry, so every ``potpie resource ...`` command against that
host was refused as a caller mistake, and ``potpie doctor`` — which asks for
resource status unconditionally — produced nothing at all. ``ledger.status`` was
the next blocker behind it.

``tests/unit/test_managed_surface_contract.py`` fixed the *client's* behaviour
when a host refuses a surface (it degrades, it says ``not_implemented``). That is
recovery, not prevention. This module is the prevention half, in two layers:

* **the manifest** (``contracts/rpc_surfaces.json``) is a committed copy of the
  set. Changing what the daemon serves now requires editing a contract file, so
  the diff a reviewer reads says "the served API changed" instead of showing one
  word inside a frozenset;
* **the cross-repo comparison** reads the managed repo's copies out of its
  source when a checkout is reachable — the served policy *and* its
  hand-transcribed mirror of this allowlist — and skips, loudly, when it is not.

The cross-repo half is deliberately a source read rather than an import: the
managed package is not installed here and installing it to compare two frozensets
would be absurd. It is also, unavoidably, only a developer-machine check. Neither
repo can fail the other's CI; what the manifest buys is that the *next* change to
either set is a deliberate one with a written-down counterpart, which is exactly
what was missing when these two silently diverged.
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
from typing import Any

import pytest

from potpie.daemon import surfaces
from potpie.daemon.client import _SURFACE_DEADLINES

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "contracts" / "rpc_surfaces.json"
MANIFEST: dict[str, Any] = json.loads(MANIFEST_PATH.read_text())
MANAGED = MANIFEST["managed_copy"]


# --- the pinned set ----------------------------------------------------------


def test_the_daemon_serves_exactly_the_pinned_surface_set() -> None:
    """Widening or narrowing the allowlist has to be written down twice."""
    pinned = frozenset(MANIFEST["surfaces"])

    added = sorted(surfaces.RPC_SURFACES - pinned)
    dropped = sorted(pinned - surfaces.RPC_SURFACES)

    assert not added and not dropped, (
        f"the daemon now serves {added} that {MANIFEST_PATH.name} does not pin, "
        f"and no longer serves {dropped} that it does. If the change is "
        f"intended, edit contracts/{MANIFEST_PATH.name} in the same commit — and "
        "tell the managed service's maintainers, because their copy of this set "
        "does not update itself."
    )


def test_the_pinned_denials_stay_denied() -> None:
    """A surface withheld on purpose must not be quietly promoted.

    ``daemon`` (process lifecycle the CLI drives locally) and ``profile`` (a
    ``str``, not a surface) are absent by decision. Pinning the decision means a
    later reader cannot mistake it for an oversight and "fix" it.
    """
    assert frozenset(MANIFEST["denied"]) == surfaces.DENIED_SURFACES
    assert not (surfaces.DENIED_SURFACES & frozenset(MANIFEST["surfaces"]))


def test_the_manifest_contract_version_tracks_the_published_one() -> None:
    """``GET /surfaces`` publishes this number; a stale pin would misdate it."""
    assert MANIFEST["contract"] == surfaces.SURFACE_CONTRACT_VERSION


@pytest.mark.parametrize("key", ["surfaces", "denied"])
def test_the_manifest_reads_as_a_canonical_list(key: str) -> None:
    """Sorted and unique, so a diff on this file is readable as a set change."""
    entries = MANIFEST[key]

    assert entries == sorted(entries), f"{key} is not sorted"
    assert len(entries) == len(set(entries)), f"{key} repeats an entry"


def test_the_clients_per_surface_deadlines_name_real_surfaces() -> None:
    """The third copy of a surface name in this repo, and the quietest one.

    ``_SURFACE_DEADLINES`` keys off surface names to give ``setup`` no
    client-side deadline. A rename that misses this table does not fail — the
    surface silently falls back to the 30s default and cold first runs start
    timing out mid-install.
    """
    unknown = sorted(frozenset(_SURFACE_DEADLINES) - surfaces.RPC_SURFACES)

    assert unknown == [], (
        f"potpie.daemon.client._SURFACE_DEADLINES keys off {unknown}, which the "
        "daemon does not serve; the entry is dead and its surface has lost its "
        "custom deadline"
    )


# --- the managed service's copy ----------------------------------------------


def _managed_checkout() -> Path | None:
    """Where the managed service's source is, or ``None`` if it is not here."""
    override = os.environ.get(MANAGED["checkout_env"])
    if override:
        return Path(override).expanduser()
    sibling = REPO_ROOT.parent / MANAGED["sibling_dir"]
    return sibling if sibling.is_dir() else None


def _managed_source(relative: str) -> Path:
    """The managed file to read, skipping (or failing) when it is unavailable.

    Absent sibling checkout: skip with the reason spelled out — this test is a
    developer-machine check and CI has no second repo. Explicitly pointed at a
    checkout that does not hold the file: fail, because someone asked for the
    comparison and is not getting it.
    """
    checkout = _managed_checkout()
    if checkout is None:
        pytest.skip(
            f"no managed-service checkout: set {MANAGED['checkout_env']} to a "
            f"clone of {MANAGED['repo']}, or place one at "
            f"{REPO_ROOT.parent / MANAGED['sibling_dir']}, to compare the two "
            "copies of the RPC allowlist"
        )
    path = checkout / relative
    if not path.is_file():
        if os.environ.get(MANAGED["checkout_env"]):
            pytest.fail(
                f"{MANAGED['checkout_env']}={checkout} does not contain "
                f"{relative}; point it at a clone of {MANAGED['repo']}"
            )
        pytest.skip(f"{checkout} is not a {MANAGED['repo']} checkout ({relative})")
    return path


def _string_members(node: ast.AST) -> frozenset[str] | None:
    """The string keys/elements of a literal collection, or ``None``.

    Handles the two shapes the managed repo uses: a ``dict`` keyed by surface
    name, and a ``frozenset({...})`` of them.
    """
    if isinstance(node, ast.Call) and node.args:
        return _string_members(node.args[0])
    if isinstance(node, ast.Dict):
        keys = node.keys
        if any(
            not isinstance(k, ast.Constant) or not isinstance(k.value, str)
            for k in keys
        ):
            return None
        return frozenset(k.value for k in keys)  # type: ignore[union-attr]
    if isinstance(node, (ast.Set, ast.List, ast.Tuple)):
        if any(
            not isinstance(e, ast.Constant) or not isinstance(e.value, str)
            for e in node.elts
        ):
            return None
        return frozenset(e.value for e in node.elts)  # type: ignore[union-attr]
    return None


def _declared_surfaces(path: Path, symbol: str) -> frozenset[str]:
    """Read ``symbol`` out of ``path`` without importing it.

    The managed package is not installed in this environment and importing it
    to read a frozenset would drag its whole dependency tree in. Parsing is also
    the honest model of what this check is: reading the other repo's source.
    """
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names = [node.target.id]
        elif isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        else:
            continue
        if symbol not in names:
            continue
        members = _string_members(node.value)
        if members is None:
            pytest.fail(
                f"{symbol} in {path} is no longer a literal this test can read; "
                "update the extractor rather than dropping the comparison"
            )
        return members
    pytest.fail(
        f"{path} no longer defines {symbol}; the managed service's allowlist "
        "moved, and this comparison is now checking nothing"
    )


@pytest.mark.parametrize(
    "source",
    MANAGED["sources"],
    ids=[entry["symbol"] for entry in MANAGED["sources"]],
)
def test_the_managed_service_agrees_on_the_surface_set(source: dict[str, str]) -> None:
    """The comparison nothing has ever made — the whole of the ``resources`` bug.

    Both copies are checked: what that service actually serves, and its
    transcription of this daemon's allowlist. The second is not redundant — a
    transcription that lags is how the first one came to be missing an entry.
    """
    managed = _declared_surfaces(_managed_source(source["path"]), source["symbol"])

    missing = sorted(surfaces.RPC_SURFACES - managed)
    extra = sorted(managed - surfaces.RPC_SURFACES)

    assert not missing and not extra, (
        f"{source['symbol']} in {MANAGED['repo']}/{source['path']} "
        f"({source['what']}) is missing {missing} and adds {extra}. A surface "
        "this daemon serves and that one does not is refused there as a caller "
        "mistake — which is how 'potpie doctor' came to be dead against managed."
    )
