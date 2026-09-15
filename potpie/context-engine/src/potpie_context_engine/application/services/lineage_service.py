"""Capture and reverse-query generation lineage (SQLite + optional graph)."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from potpie_context_engine.application.services.lineage_store import (
    LineageStore,
    SpanHit,
    default_lineage_db_path,
)
from potpie_context_engine.core.identity import get_identity, mint_entity_key
from potpie_context_engine.core.ports.claim_query import (
    ClaimQueryFilter,
    ClaimQueryPort,
)

logger = logging.getLogger(__name__)

GraphRecorder = Callable[[str, str, Mapping[str, Any], Mapping[str, Any]], Any]

_WHY_PREDICATES = (
    "GENERATED_FROM",
    "IMPLEMENTS",
    "IN_SESSION",
    "DERIVED_FROM",
    "MODIFIES",
    "USED_CONTEXT",
)


def _span_hash8(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:8]


def mint_prompt_key(text: str) -> str:
    return mint_entity_key(get_identity("PromptTurn"), content=text)


def mint_spec_key(text: str) -> str:
    return mint_entity_key(get_identity("SpecRequirement"), content=text)


def mint_session_key(*, harness: str, session_id: str) -> str:
    return mint_entity_key(
        get_identity("GenerationSession"),
        name=session_id or "default",
        extra_segments=(harness or "unknown",),
    )


def mint_code_asset_key(
    *,
    repo: str,
    path: str,
    line_start: int,
    line_end: int,
    span_text: str,
) -> str:
    normalized = path.replace("\\", "/").lstrip("./")
    return mint_entity_key(
        get_identity("CodeAsset"),
        name=f"l{line_start}-l{line_end}-{_span_hash8(span_text)}",
        extra_segments=(repo or "local", normalized),
    )


def parse_line_range(raw: str | None) -> tuple[int, int] | None:
    if not raw:
        return None
    text = raw.strip().replace(":", "-")
    if "-" not in text:
        start = int(text)
        return start, start
    left, _, right = text.partition("-")
    start, end = int(left), int(right)
    if end < start:
        start, end = end, start
    return start, end


def line_range_of_snippet(file_text: str, snippet: str) -> tuple[int, int] | None:
    if not snippet:
        return None
    idx = file_text.find(snippet)
    if idx < 0:
        return None
    start = file_text[:idx].count("\n") + 1
    end = start + snippet.count("\n")
    return start, max(end, start)


def _read_span_text(path: Path, line_start: int, line_end: int) -> str:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""
    start = max(line_start, 1)
    end = min(line_end, len(lines))
    if end < start:
        return ""
    return "\n".join(lines[start - 1 : end])


@dataclass
class LineageService:
    store: LineageStore
    pot_id: str = ""
    record_graph: GraphRecorder | None = None
    claim_query: ClaimQueryPort | None = None
    fail_open: bool = True

    @classmethod
    def for_pot(
        cls,
        pot_id: str,
        *,
        home: Path | None = None,
        record_graph: GraphRecorder | None = None,
        claim_query: ClaimQueryPort | None = None,
        fail_open: bool = True,
    ) -> LineageService:
        return cls(
            store=LineageStore(default_lineage_db_path(pot_id, home=home)),
            pot_id=pot_id,
            record_graph=record_graph,
            claim_query=claim_query,
            fail_open=fail_open,
        )

    def remember_prompt(
        self,
        *,
        prompt: str,
        harness: str = "unknown",
        session_id: str = "default",
        used_context_keys: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        prompt_key = mint_prompt_key(prompt)
        prompt_hash = prompt_key.rsplit(":", 1)[-1]
        session_key = mint_session_key(harness=harness, session_id=session_id)
        self.store.put_payload(hash_=prompt_hash, kind="prompt", text=prompt)
        self.store.put_session(
            session_key=session_key, harness=harness, session_id=session_id
        )
        self.store.remember_prompt(session_key=session_key, prompt_hash=prompt_hash)
        graph_error = self._record(
            "prompt_turn",
            summary=prompt[:240],
            details={
                "prompt_key": prompt_key,
                "session_key": session_key,
                "harness": harness,
                "session_id": session_id,
                "used_context_keys": list(used_context_keys),
            },
        )
        return {
            "ok": True,
            "prompt_key": prompt_key,
            "prompt_hash": prompt_hash,
            "session_key": session_key,
            "graph_error": graph_error,
        }

    def capture(
        self,
        *,
        path: str,
        line_start: int,
        line_end: int,
        prompt: str | None = None,
        spec: str | None = None,
        harness: str = "unknown",
        session_id: str = "default",
        repo: str = "local",
        span_text: str | None = None,
    ) -> dict[str, Any]:
        session_key = mint_session_key(harness=harness, session_id=session_id)
        self.store.put_session(
            session_key=session_key, harness=harness, session_id=session_id
        )

        prompt_key = None
        prompt_hash = None
        if prompt and prompt.strip():
            remembered = self.remember_prompt(
                prompt=prompt, harness=harness, session_id=session_id
            )
            prompt_key = remembered["prompt_key"]
            prompt_hash = remembered["prompt_hash"]
        else:
            prompt_hash = self.store.latest_prompt_hash(session_key)
            if prompt_hash:
                prompt_key = f"prompt:{prompt_hash}"

        spec_key = None
        spec_hash = None
        if spec and spec.strip():
            spec_key = mint_spec_key(spec)
            spec_hash = spec_key.rsplit(":", 1)[-1]
            self.store.put_payload(hash_=spec_hash, kind="spec", text=spec)
            self._record(
                "spec_requirement",
                summary=spec[:240],
                details={
                    "spec_key": spec_key,
                    "prompt_key": prompt_key,
                    "status": "draft",
                },
            )

        file_path = Path(path)
        text = span_text
        if text is None:
            text = _read_span_text(file_path, line_start, line_end)
        code_asset_key = mint_code_asset_key(
            repo=repo,
            path=str(path),
            line_start=line_start,
            line_end=line_end,
            span_text=text or f"{path}:{line_start}-{line_end}",
        )
        normalized = path.replace("\\", "/")
        self.store.put_span(
            path=normalized,
            line_start=line_start,
            line_end=line_end,
            code_asset_key=code_asset_key,
            prompt_hash=prompt_hash,
            spec_hash=spec_hash,
            session_key=session_key,
        )
        graph_error = None
        if prompt_key:
            graph_error = self._record(
                "generation_link",
                summary=f"{normalized}:{line_start}-{line_end}",
                details={
                    "code_asset_key": code_asset_key,
                    "prompt_key": prompt_key,
                    "spec_key": spec_key,
                    "session_key": session_key,
                    "path": normalized,
                    "line_start": line_start,
                    "line_end": line_end,
                },
                scope={"repo": repo, "file_path": normalized},
            )
        return {
            "ok": True,
            "path": normalized,
            "line_start": line_start,
            "line_end": line_end,
            "code_asset_key": code_asset_key,
            "prompt_key": prompt_key,
            "prompt_hash": prompt_hash,
            "spec_key": spec_key,
            "spec_hash": spec_hash,
            "session_key": session_key,
            "graph_error": graph_error,
        }

    def why(self, *, path: str, line_start: int, line_end: int) -> dict[str, Any]:
        """SQLite span lookup, then optional provenance claim walk (plan why-query)."""
        hits = self.store.overlapping_spans(
            path=path, line_start=line_start, line_end=line_end
        )
        matches = [_hit_payload(hit) for hit in hits]
        graph_error: str | None = None
        if self.claim_query is not None and matches:
            try:
                for match in matches:
                    match["claims"] = self._walk_claims(match.get("code_asset_key"))
            except Exception as exc:  # noqa: BLE001 - why must fail open on graph
                logger.debug("lineage why graph walk failed: %s", exc)
                graph_error = str(exc)
                if not self.fail_open:
                    raise
        return {
            "ok": True,
            "path": path.replace("\\", "/"),
            "line_start": line_start,
            "line_end": line_end,
            "matches": matches,
            "graph_error": graph_error,
        }

    def _walk_claims(self, code_asset_key: str | None) -> list[dict[str, Any]]:
        """Walk live provenance claims for a CodeAsset key (Trustgraph-style reverse)."""
        if not code_asset_key or self.claim_query is None:
            return []
        pot = self.pot_id or "default"
        rows = list(
            self.claim_query.find_claims(
                ClaimQueryFilter(
                    pot_id=pot,
                    predicate_in=_WHY_PREDICATES,
                    subject_key_in=(code_asset_key,),
                    include_invalidated=False,
                    limit=64,
                )
            )
        )
        rows += list(
            self.claim_query.find_claims(
                ClaimQueryFilter(
                    pot_id=pot,
                    predicate_in=_WHY_PREDICATES,
                    object_key_in=(code_asset_key,),
                    include_invalidated=False,
                    limit=64,
                )
            )
        )
        seen: set[tuple[str, str, str]] = set()
        out: list[dict[str, Any]] = []
        for row in rows:
            key = (row.predicate, row.subject_key, row.object_key)
            if key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    "predicate": row.predicate,
                    "subject_key": row.subject_key,
                    "object_key": row.object_key,
                    "claim_key": row.claim_key,
                    "source_ref": row.source_ref,
                }
            )
        return out

    def _record(
        self,
        record_type: str,
        *,
        summary: str,
        details: Mapping[str, Any],
        scope: Mapping[str, Any] | None = None,
    ) -> str | None:
        if self.record_graph is None:
            return None
        try:
            self.record_graph(record_type, summary, dict(details), dict(scope or {}))
            return None
        except Exception as exc:  # noqa: BLE001 - capture must fail open
            logger.debug("lineage graph record failed: %s", exc)
            if self.fail_open:
                return str(exc)
            raise


def _hit_payload(hit: SpanHit) -> dict[str, Any]:
    return {
        "path": hit.path,
        "line_start": hit.line_start,
        "line_end": hit.line_end,
        "code_asset_key": hit.code_asset_key,
        "prompt_hash": hit.prompt_hash,
        "spec_hash": hit.spec_hash,
        "session_key": hit.session_key,
        "prompt": hit.prompt_text,
        "spec": hit.spec_text,
        "created_at": hit.created_at,
    }


__all__ = [
    "LineageService",
    "line_range_of_snippet",
    "mint_code_asset_key",
    "mint_prompt_key",
    "mint_session_key",
    "mint_spec_key",
    "parse_line_range",
]
