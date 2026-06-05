"""CODEOWNERS scanner (rebuild plan P4 / F2 fix).

The proper POC's F2 failure: the LLM extractor reading a body like
``"@alice owns this"`` from ``apps/auth/CODEOWNERS`` had no way to
recover the service scope (the path ``apps/auth/`` *is* the scope) and
emitted ``(component:unknown) -[OWNED_BY]-> (person:alice)``.

This scanner closes the gap deterministically:

1. The CODEOWNERS path itself becomes the default scope (e.g.
   ``apps/auth/CODEOWNERS`` → ``service:auth``).
2. Each rule's pattern path is parsed for an *additional* scope (e.g.
   ``/services/users/`` rule under a root CODEOWNERS scopes to
   ``service:users``); rule-level scope overrides file-level scope when
   present.
3. Owners (``@user`` / ``@org/team`` / ``email``) are minted into stable
   ``person:`` / ``team:`` keys via :mod:`domain.identity`.
4. One ``OWNED_BY`` :class:`ScannerClaim` per ``(service|repo) × owner``
   pair, with ``evidence_strength="deterministic"`` so the singleton-
   predicate machinery in :mod:`domain.singleton_predicates` supersedes
   stale ownerships automatically.

CODEOWNERS lives at one of three canonical locations (per GitHub docs):
``CODEOWNERS``, ``.github/CODEOWNERS``, ``docs/CODEOWNERS``. We accept
nested files too (``apps/<svc>/CODEOWNERS``) which GitHub does not, but
many monorepos use as a convention; nested files give us *better*
scope, so they're a feature here.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

from domain.identity import IdentityError, get_identity, mint_entity_key
from domain.path_scope import PathScope, derive_scope
from domain.ports.config_scanner import (
    ConfigFileRef,
    ConfigSourceScannerCapability,
    ScanResult,
    ScannerClaim,
    ScannerEntity,
)

logger = logging.getLogger(__name__)


_FILENAME_PATTERN = re.compile(r"(^|/)CODEOWNERS$", re.IGNORECASE)
_OWNER_USER_RE = re.compile(r"^@([A-Za-z0-9][A-Za-z0-9_\-]*)$")
_OWNER_TEAM_RE = re.compile(
    r"^@([A-Za-z0-9][A-Za-z0-9_\-]*)/([A-Za-z0-9][A-Za-z0-9_\-]*)$"
)
_OWNER_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")


@dataclass(frozen=True, slots=True)
class _ParsedOwner:
    """One owner mention resolved to a canonical entity_key.

    ``label`` is the ontology label (``Person`` / ``Team``); ``raw``
    keeps the original string for debug / fact-rendering.
    """

    entity_key: str
    label: str
    raw: str
    display_name: str


@dataclass(frozen=True, slots=True)
class _CodeownersRule:
    """One non-comment line from a CODEOWNERS file."""

    line_no: int
    pattern: str
    owners: tuple[_ParsedOwner, ...]


class CodeownersScanner:
    """Deterministic ``Service -[OWNED_BY]-> Person|Team`` emitter."""

    SCANNER_KIND = "codeowners"
    PREDICATE = "OWNED_BY"

    def __init__(self, *, source_system: str = "codeowners") -> None:
        # ``source_system`` is recorded on every emitted claim so the
        # belief-derivation layer can apply the right authority weight.
        self._source_system = source_system

    # ------------------------------------------------------------------
    # ConfigSourceScannerPort
    # ------------------------------------------------------------------
    def kind(self) -> str:
        return self.SCANNER_KIND

    def capabilities(self) -> ConfigSourceScannerCapability:
        return ConfigSourceScannerCapability(
            kind=self.SCANNER_KIND,
            description=(
                "Parses CODEOWNERS files; emits deterministic Service OWNED_BY "
                "Person|Team claims with path-aware scope (F2 fix)."
            ),
            handles_file_patterns=(
                "CODEOWNERS",
                ".github/CODEOWNERS",
                "docs/CODEOWNERS",
            ),
            emits_predicates=(self.PREDICATE,),
        )

    def handles(self, file_ref: ConfigFileRef) -> bool:
        return _FILENAME_PATTERN.search(file_ref.path) is not None

    def list_files(
        self, *, repo_name: str, working_tree_paths: Iterable[str]
    ) -> Iterable[str]:
        del repo_name  # repo name is host-side concern; we match on path shape
        return [p for p in working_tree_paths if _FILENAME_PATTERN.search(p)]

    def parse_to_claims(self, file_ref: ConfigFileRef) -> ScanResult:
        warnings: list[str] = []
        try:
            rules = _parse_codeowners(
                content=file_ref.content,
                warnings=warnings,
            )
        except Exception as exc:  # pragma: no cover - parser guard
            logger.exception("CODEOWNERS parser failed: %s", exc)
            return ScanResult(warnings=(f"parser-exception: {exc}",))

        file_scope = derive_scope(file_ref.path)
        observed_at = file_ref.observed_at or datetime.now(tz=timezone.utc)
        valid_at = observed_at  # commit timestamp would be better; host can override

        repo_key = _repo_key(file_ref.repo_name)
        entities: list[ScannerEntity] = []
        claims: list[ScannerClaim] = []
        seen_entities: set[str] = set()
        seen_edges: set[tuple[str, str]] = set()

        for rule in rules:
            rule_scope = _rule_pattern_scope(rule.pattern)
            scope = file_scope.merged_with(rule_scope)
            subject = _subject_for_scope(scope=scope, repo_key=repo_key)
            if subject is None:
                warnings.append(
                    f"codeowners:{file_ref.path}:line {rule.line_no}: "
                    f"no scope and no repo name — skipping rule '{rule.pattern}'"
                )
                continue

            subject_key, subject_label, subject_name = subject
            if subject_key not in seen_entities:
                entities.append(
                    ScannerEntity(
                        entity_key=subject_key,
                        label=subject_label,
                        name=subject_name,
                        properties={"derived_from": "codeowners-scanner"},
                    )
                )
                seen_entities.add(subject_key)

            for owner in rule.owners:
                if owner.entity_key not in seen_entities:
                    entities.append(
                        ScannerEntity(
                            entity_key=owner.entity_key,
                            label=owner.label,
                            name=owner.display_name,
                            properties={"derived_from": "codeowners-scanner"},
                        )
                    )
                    seen_entities.add(owner.entity_key)

                edge_key = (subject_key, owner.entity_key)
                if edge_key in seen_edges:
                    # Same owner appeared twice for same subject across
                    # rules (e.g. a more-specific override). Keep the
                    # earlier (top-of-file) wins — later identical
                    # entries add nothing.
                    continue
                seen_edges.add(edge_key)

                source_ref = _source_ref(
                    repo=file_ref.repo_name,
                    path=file_ref.path,
                    line=rule.line_no,
                )
                fact = _render_fact(
                    subject_name=subject_name or subject_key,
                    owner=owner,
                )
                claims.append(
                    ScannerClaim(
                        subject_key=subject_key,
                        predicate=self.PREDICATE,
                        object_key=owner.entity_key,
                        source_ref=source_ref,
                        source_system=self._source_system,
                        fact=fact,
                        valid_at=valid_at,
                        evidence_strength="deterministic",
                        properties={
                            "rule_pattern": rule.pattern,
                            "rule_line": rule.line_no,
                            "owner_kind": owner.label.lower(),
                            "scope_service": scope.service,
                            "scope_environment": scope.environment,
                            "observed_at": observed_at,
                        },
                    )
                )

        return ScanResult(
            entities=tuple(entities),
            claims=tuple(claims),
            warnings=tuple(warnings),
        )


# ---------------------------------------------------------------------------
# Parsing helpers (module-private)
# ---------------------------------------------------------------------------


def _parse_codeowners(*, content: str, warnings: list[str]) -> list[_CodeownersRule]:
    rules: list[_CodeownersRule] = []
    for idx, raw_line in enumerate(content.splitlines(), start=1):
        stripped = raw_line.split("#", 1)[0].strip()
        if not stripped:
            continue
        tokens = stripped.split()
        if len(tokens) < 2:
            warnings.append(
                f"codeowners:line {idx}: rule '{raw_line.rstrip()}' has no owners — skipping"
            )
            continue
        pattern, *owner_tokens = tokens
        parsed_owners: list[_ParsedOwner] = []
        for tok in owner_tokens:
            owner = _parse_owner(tok)
            if owner is None:
                warnings.append(
                    f"codeowners:line {idx}: unrecognised owner token {tok!r} — skipping"
                )
                continue
            parsed_owners.append(owner)
        if not parsed_owners:
            continue
        rules.append(
            _CodeownersRule(
                line_no=idx,
                pattern=pattern,
                owners=tuple(parsed_owners),
            )
        )
    return rules


def _parse_owner(token: str) -> _ParsedOwner | None:
    user_match = _OWNER_USER_RE.match(token)
    if user_match:
        username = user_match.group(1).lower()
        try:
            spec = get_identity("Person")
            if spec is None:
                return None
            key = mint_entity_key(spec, name=username)
        except IdentityError:
            return None
        return _ParsedOwner(
            entity_key=key,
            label="Person",
            raw=token,
            display_name=user_match.group(1),
        )

    team_match = _OWNER_TEAM_RE.match(token)
    if team_match:
        org, team = team_match.group(1).lower(), team_match.group(2).lower()
        try:
            spec = get_identity("Team")
            if spec is None:
                return None
            key = mint_entity_key(spec, name=f"{org}-{team}")
        except IdentityError:
            return None
        return _ParsedOwner(
            entity_key=key,
            label="Team",
            raw=token,
            display_name=f"@{org}/{team}",
        )

    if _OWNER_EMAIL_RE.match(token):
        # Slug the local-part; full email lives on the Person entity's
        # properties at upsert time (caller's job).
        local = token.split("@", 1)[0].lower()
        try:
            spec = get_identity("Person")
            if spec is None:
                return None
            key = mint_entity_key(spec, name=local)
        except IdentityError:
            return None
        return _ParsedOwner(
            entity_key=key,
            label="Person",
            raw=token,
            display_name=token,
        )

    return None


def _rule_pattern_scope(pattern: str) -> PathScope:
    """Best-effort scope extraction from a CODEOWNERS pattern string.

    CODEOWNERS patterns are gitignore-style; we strip leading ``/`` and
    glob wildcards, then run :func:`derive_scope` over the residue. A
    pattern of ``*`` or ``**`` yields an empty scope (file-level scope
    still applies).
    """
    if not pattern or pattern in ("*", "**", "/*", "/**", "/"):
        return PathScope()
    cleaned = pattern.strip().lstrip("/")
    # Replace glob wildcards with empty so the path matcher sees a real path
    cleaned = re.sub(r"\*+", "", cleaned)
    cleaned = cleaned.replace("//", "/").strip("/")
    if not cleaned:
        return PathScope()
    return derive_scope(cleaned)


def _subject_for_scope(
    *, scope: PathScope, repo_key: str | None
) -> tuple[str, str, str | None] | None:
    """Return (subject_key, label, display_name) for the rule subject.

    Prefer the most specific scope: service > repo. If neither is
    available we have no subject and return ``None`` — caller emits a
    warning and skips the rule.
    """
    if scope.service:
        try:
            spec = get_identity("Service")
            if spec is None:
                return None
            key = mint_entity_key(spec, name=scope.service)
            return (key, "Service", scope.service)
        except IdentityError:
            return None
    if repo_key:
        return (repo_key, "Repository", None)
    return None


def _repo_key(repo_name: str | None) -> str | None:
    if not repo_name:
        return None
    spec = get_identity("Repository")
    if spec is None:
        return None
    try:
        return mint_entity_key(spec, name=repo_name)
    except IdentityError:
        return None


def _source_ref(*, repo: str | None, path: str, line: int) -> str:
    repo_seg = repo or "unknown-repo"
    return f"codeowners:{repo_seg}:{path}:L{line}"


def _render_fact(*, subject_name: str, owner: _ParsedOwner) -> str:
    return f"{subject_name} is owned by {owner.display_name}"


__all__ = ["CodeownersScanner"]
