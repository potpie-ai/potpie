"""Lineage commands: ``potpie why`` and ``potpie lineage capture``."""

from __future__ import annotations

from pathlib import Path

import typer

from potpie.cli.commands._common import (
    contract,
    emit,
    get_engine_client,
    get_root_runtime,
    resolve_pot_id,
    run_engine_operation,
)
from potpie_context_engine.application.services.lineage_service import (
    LineageService,
    parse_line_range,
)
from potpie_context_engine.requests import ReadRequest as EngineReadRequest
from potpie_context_engine.requests import RecordRequest as EngineRecordRequest

lineage_app = typer.Typer(help="Capture and query code↔prompt generation lineage.")


def _read_prompt(prompt: str | None, prompt_file: str | None) -> str | None:
    if prompt_file:
        if prompt_file == "-":
            return typer.get_text_stream("stdin").read()
        return Path(prompt_file).read_text(encoding="utf-8")
    return prompt


def _graph_recorder(pot: str | None):
    def _record(
        record_type: str,
        summary: str,
        details: dict,
        scope: dict,
    ) -> object:
        return run_engine_operation(
            get_engine_client(pot).record(
                EngineRecordRequest(
                    record_type=record_type,
                    summary=summary,
                    details=details,
                    scope=scope,
                )
            )
        )

    return _record


def _service(pot: str | None, *, fail_open: bool) -> tuple[str, LineageService]:
    pot_id = resolve_pot_id(get_root_runtime(), pot)
    return pot_id, LineageService.for_pot(
        pot_id,
        record_graph=_graph_recorder(pot),
        fail_open=fail_open,
    )


def _attach_graph_claims(result: dict, *, pot: str | None, path: str) -> None:
    """After SQLite span hits, walk provenance claims via the lineage reader (fail-open)."""
    try:
        graph = run_engine_operation(
            get_engine_client(pot).read(
                EngineReadRequest(
                    subgraph="provenance",
                    view="lineage",
                    scope={"path": path.replace("\\", "/")},
                    limit=32,
                    detail="compact",
                    relations="summary",
                )
            )
        )
        payload = getattr(graph, "result", None)
        if payload is None and isinstance(graph, dict):
            payload = graph.get("result") or graph
        items: list = []
        if payload is not None:
            if isinstance(payload, dict):
                items = list(payload.get("items") or [])
            elif hasattr(payload, "items"):
                items = list(getattr(payload, "items") or [])
        elif hasattr(graph, "items"):
            # GraphReadResult exposes items on the envelope body directly.
            items = list(getattr(graph, "items") or [])
        elif isinstance(graph, dict):
            items = list(graph.get("items") or [])
        # Prefer attaching claim summaries onto SQLite matches by code_asset_key.
        by_key: dict[str, list[dict]] = {}
        compact_items: list[dict] = []
        def _row_from_item(raw: object) -> dict | None:
            if isinstance(raw, dict):
                item = raw
            elif hasattr(raw, "__dict__"):
                item = {
                    "entity_key": getattr(raw, "entity_key", None),
                    "summary": getattr(raw, "summary", None),
                    "relation_predicates": getattr(raw, "relation_predicates", None),
                    "related_keys": getattr(raw, "related_keys", None),
                    "relations": getattr(raw, "relations", None),
                }
            else:
                return None
            predicates = list(item.get("relation_predicates") or [])
            related = list(item.get("related_keys") or [])
            relations = item.get("relations") or []
            if not predicates and relations:
                predicates = [
                    str(rel.get("predicate") or rel.get("type") or "")
                    for rel in relations
                    if isinstance(rel, dict)
                ]
            if not related and relations:
                seen_related: set[str] = set()
                for rel in relations:
                    if not isinstance(rel, dict):
                        continue
                    for key in ("related_key", "to_key", "from_key"):
                        val = rel.get(key)
                        if isinstance(val, str) and val:
                            seen_related.add(val)
                related = sorted(seen_related)
            return {
                "entity_key": item.get("entity_key"),
                "relation_predicates": predicates,
                "related_keys": related,
                "summary": item.get("summary"),
            }

        for item in items:
            row = _row_from_item(item)
            if row is None:
                continue
            compact_items.append(row)
            key = row.get("entity_key")
            if isinstance(key, str) and key.startswith("code:"):
                by_key.setdefault(key, []).append(row)
        for match in result.get("matches") or []:
            code_key = match.get("code_asset_key")
            if code_key and code_key in by_key:
                match["graph"] = by_key[code_key]
            elif code_key:
                # Still surface related path-scoped items when keys differ by span hash.
                match["graph"] = [
                    row
                    for row in compact_items
                    if code_key in (row.get("related_keys") or [])
                    or code_key == row.get("entity_key")
                ]
        result["graph_items"] = compact_items
    except Exception as exc:  # noqa: BLE001 - why must never fail closed on graph
        result["graph_error"] = str(exc)


def register(root: typer.Typer) -> None:
    @root.command("why")
    def why(
        path: str = typer.Argument(..., help="File path to look up."),
        lines: str = typer.Option(..., "--lines", help="Line range, e.g. 12-40."),
        pot: str = typer.Option(None, "--pot"),
    ) -> None:
        """Trace a file+line range back to the prompt and spec that produced it."""
        with contract():
            parsed = parse_line_range(lines)
            if parsed is None:
                raise typer.BadParameter("use --lines START-END")
            start, end = parsed
            pot_id, service = _service(pot, fail_open=True)
            result = service.why(path=path, line_start=start, line_end=end)
            result["pot_id"] = pot_id
            _attach_graph_claims(result, pot=pot, path=path)
            matches = result.get("matches") or []
            if not matches:
                human = f"no lineage for {path}:{start}-{end}"
            else:
                first = matches[0]
                human = (
                    f"{path}:{start}-{end}\n"
                    f"  session {first.get('session_key')}\n"
                    f"  prompt  {first.get('prompt_hash')}\n"
                    f"  spec    {first.get('spec_hash')}"
                )
                graph = first.get("graph") or []
                if graph:
                    preds = sorted(
                        {
                            p
                            for row in graph
                            for p in (row.get("relation_predicates") or [])
                        }
                    )
                    if preds:
                        human += f"\n  graph   {', '.join(preds)}"
            emit(result, human=human)

    root.add_typer(lineage_app, name="lineage")


@lineage_app.command("capture")
def lineage_capture(
    path: str = typer.Option(None, "--path", help="File that was written or edited."),
    lines: str = typer.Option(None, "--lines", help="Line range, e.g. 12-40."),
    prompt: str = typer.Option(None, "--prompt"),
    prompt_file: str = typer.Option(None, "--prompt-file"),
    spec: str = typer.Option(None, "--spec"),
    session: str = typer.Option("default", "--session"),
    harness: str = typer.Option("unknown", "--harness"),
    repo: str = typer.Option("local", "--repo"),
    remember_prompt: bool = typer.Option(
        False, "--remember-prompt", help="Store the prompt without a file span."
    ),
    fail_open: bool = typer.Option(
        True, "--fail-open/--no-fail-open", help="Never raise on capture failure."
    ),
    pot: str = typer.Option(None, "--pot"),
) -> None:
    """Write SQLite lineage (and graph claims) for a prompt and/or code span."""
    with contract():
        try:
            pot_id, service = _service(pot, fail_open=fail_open)
            prompt_text = _read_prompt(prompt, prompt_file)
            if remember_prompt or (prompt_text and not path):
                if not prompt_text or not prompt_text.strip():
                    raise typer.BadParameter("prompt text is required")
                result = service.remember_prompt(
                    prompt=prompt_text, harness=harness, session_id=session
                )
            else:
                if not path:
                    raise typer.BadParameter("--path is required unless --remember-prompt")
                parsed = parse_line_range(lines)
                if parsed is None:
                    file_path = Path(path)
                    if file_path.is_file():
                        n = len(file_path.read_text(encoding="utf-8").splitlines()) or 1
                        parsed = (1, n)
                    else:
                        parsed = (1, 1)
                start, end = parsed
                result = service.capture(
                    path=path,
                    line_start=start,
                    line_end=end,
                    prompt=prompt_text,
                    spec=spec,
                    harness=harness,
                    session_id=session,
                    repo=repo,
                )
            result["pot_id"] = pot_id
            emit(
                result,
                human=(
                    f"captured {result.get('path') or 'prompt'} "
                    f"session={result.get('session_key')} "
                    f"prompt={result.get('prompt_hash')}"
                ),
            )
        except Exception as exc:  # noqa: BLE001
            if fail_open:
                emit(
                    {"ok": False, "fail_open": True, "error": str(exc)},
                    human=f"lineage capture failed open: {exc}",
                )
                return
            raise


__all__ = ["lineage_app", "register"]
