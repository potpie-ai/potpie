"""Bulk mutation payload helpers for graph CLI commands."""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from typing import Any

def _load_json(file: str | None) -> dict:
    """Load a mutation payload from a file or stdin."""
    raw = _read_payload_text(file)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in mutation payload: {exc}") from exc


def _read_payload_text(file: str | None) -> str:
    if file:
        try:
            with open(file, encoding="utf-8") as fh:
                raw = fh.read()
        except OSError as exc:
            raise ValueError(f"cannot read mutation file {file!r}: {exc}") from exc
    else:
        raw = sys.stdin.read()
    if not raw.strip():
        raise ValueError(
            "empty mutation payload (provide --file or pipe JSON on stdin)"
        )
    return raw


def _load_bulk_mutation_payloads(file: str | None) -> list[dict[str, Any]]:
    raw = _read_payload_text(file)
    stripped = raw.strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return _load_bulk_ndjson(stripped)
    return [_normalize_bulk_payload(parsed, context="mutation payload")]


def _load_bulk_ndjson(raw: str) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    pending_ops: list[Mapping[str, Any]] = []
    for line_no, line in enumerate(raw.splitlines(), start=1):
        clean = line.strip()
        if not clean or clean.startswith("#"):
            continue
        try:
            parsed = json.loads(clean)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON on NDJSON line {line_no}: {exc}") from exc
        if _is_batch_payload(parsed):
            if pending_ops:
                payloads.append({"operations": list(pending_ops)})
                pending_ops = []
            payloads.append(
                _normalize_bulk_payload(parsed, context=f"NDJSON line {line_no}")
            )
            continue
        if not isinstance(parsed, Mapping):
            raise ValueError(f"NDJSON line {line_no} must be a JSON object")
        pending_ops.append(dict(parsed))
    if pending_ops:
        payloads.append({"operations": list(pending_ops)})
    if not payloads:
        raise ValueError("bulk mutation input did not contain any operations")
    return payloads


def _normalize_bulk_payload(value: Any, *, context: str) -> dict[str, Any]:
    if isinstance(value, list):
        operations = value
        metadata: dict[str, Any] = {}
    elif isinstance(value, Mapping):
        if "operations" not in value:
            operations = [dict(value)]
            metadata = {}
        else:
            raw_ops = value.get("operations")
            if not isinstance(raw_ops, list):
                raise ValueError(f"{context} field 'operations' must be a list")
            operations = raw_ops
            metadata = {str(k): v for k, v in value.items() if k != "operations"}
    else:
        raise ValueError(f"{context} must be a JSON object or array")

    if not operations:
        raise ValueError(f"{context} contains no operations")
    clean_ops: list[dict[str, Any]] = []
    for index, op in enumerate(operations, start=1):
        if not isinstance(op, Mapping):
            raise ValueError(f"{context} operation {index} must be a JSON object")
        clean_ops.append(dict(op))
    metadata["operations"] = clean_ops
    return metadata


def _is_batch_payload(value: Any) -> bool:
    return isinstance(value, Mapping) and isinstance(value.get("operations"), list)


def _build_bulk_chunks(
    payloads: list[dict[str, Any]],
    *,
    chunk_size: int,
    idempotency_key: str | None,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    chunk_index = 1
    for payload_index, payload in enumerate(payloads, start=1):
        metadata = {k: v for k, v in payload.items() if k != "operations"}
        operations = list(payload["operations"])
        chunk_count = (len(operations) + chunk_size - 1) // chunk_size
        base_idempotency = (
            idempotency_key
            or str(metadata.get("idempotency_key") or "").strip()
            or None
        )
        for offset in range(0, len(operations), chunk_size):
            ops = operations[offset : offset + chunk_size]
            chunk_payload = dict(metadata)
            if base_idempotency:
                if chunk_count > 1 or len(payloads) > 1:
                    chunk_payload["idempotency_key"] = (
                        f"{base_idempotency}:chunk-{chunk_index:04d}"
                    )
                else:
                    chunk_payload["idempotency_key"] = base_idempotency
            chunk_payload["operations"] = ops
            chunks.append(
                {
                    "index": chunk_index,
                    "source_payload_index": payload_index,
                    "operation_count": len(ops),
                    "operations": ops,
                    "idempotency_key": chunk_payload.get("idempotency_key"),
                    "payload": chunk_payload,
                }
            )
            chunk_index += 1
    if not chunks:
        raise ValueError("bulk mutation input did not contain any operations")
    return chunks


def _new_bulk_run_payload(
    *,
    pot_id: str,
    chunks_total: int,
    operations_total: int,
    chunk_size: int,
    dry_run: bool,
    start_chunk: int,
    manifest: str | None,
) -> dict[str, Any]:
    return {
        "ok": True,
        "status": "running",
        "pot_id": pot_id,
        "chunks_total": chunks_total,
        "chunks_attempted": 0,
        "chunks_validated": 0,
        "chunks_committed": 0,
        "operations_total": operations_total,
        "operations_attempted": 0,
        "operations_validated": 0,
        "operations_committed": 0,
        "chunk_size": chunk_size,
        "dry_run": dry_run,
        "start_chunk": start_chunk,
        "manifest": manifest,
        "chunks": [],
        "issues": [],
    }


def _bulk_skipped_chunk(chunk: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "index": chunk["index"],
        "source_payload_index": chunk["source_payload_index"],
        "operation_count": chunk["operation_count"],
        "idempotency_key": chunk.get("idempotency_key"),
        "ok": True,
        "status": "skipped",
    }


def _bulk_chunk_entry(chunk: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "index": chunk["index"],
        "source_payload_index": chunk["source_payload_index"],
        "operation_count": chunk["operation_count"],
        "idempotency_key": chunk.get("idempotency_key"),
        "ok": False,
        "status": "pending",
    }


def _bulk_proposal_summary(result: Any) -> dict[str, Any]:
    out = {
        "ok": result.ok,
        "plan_id": result.plan_id,
        "status": result.status,
        "risk": result.risk,
        "auto_applicable": result.auto_applicable,
        "issues": list(result.issues),
        "rejected_operations": list(result.rejected_operations),
        "review_required_operations": list(result.review_required_operations),
        "claim_keys": list(result.claim_keys),
    }
    if result.diff:
        out["diff"] = result.diff.to_dict()
    return out


def _bulk_commit_summary(result: Any) -> dict[str, Any]:
    out = {
        "ok": result.ok,
        "plan_id": result.plan_id,
        "status": result.status,
        "risk": result.risk,
        "mutation_id": result.mutation_id,
        "detail": result.detail,
        "claim_keys": list(result.claim_keys),
    }
    if result.diff:
        out["diff"] = result.diff.to_dict()
    return out


def _bulk_issues_from_proposal(
    result: Any, chunk_index: int
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for issue in result.issues or ():
        if isinstance(issue, Mapping):
            item = dict(issue)
        else:
            item = {"code": "proposal_issue", "message": str(issue)}
        item.setdefault("severity", "error")
        item["chunk"] = chunk_index
        issues.append(item)
    if not issues:
        issues.append(
            {
                "code": str(result.status or "proposal_failed"),
                "message": "chunk proposal failed",
                "severity": "error",
                "chunk": chunk_index,
            }
        )
    return issues


def _bulk_run_status(
    run: Mapping[str, Any],
    *,
    dry_run: bool,
    ok: bool,
) -> str:
    if not ok:
        if run.get("chunks_committed") or run.get("chunks_validated"):
            return "partial"
        return "failed"
    return "validated" if dry_run else "committed"


def _bulk_next_action(run: Mapping[str, Any]) -> str | None:
    if run.get("ok"):
        if run.get("dry_run"):
            return "Rerun without --dry-run to commit the proposed chunks."
        return None
    for issue in run.get("issues") or ():
        if isinstance(issue, Mapping) and issue.get("code") == "approval_required":
            return "Review the chunk plan, then rerun with --approved-by when policy allows."
    return "Inspect the failed chunk, fix the mutation input, and rerun with --start-chunk if earlier chunks succeeded."


def _bulk_human(run: Mapping[str, Any]) -> str:
    status = run.get("status")
    attempted = run.get("chunks_attempted", 0)
    total = run.get("chunks_total", 0)
    committed = run.get("chunks_committed", 0)
    validated = run.get("chunks_validated", 0)
    ops_committed = run.get("operations_committed", 0)
    ops_validated = run.get("operations_validated", 0)
    lines = [
        (
            f"{status}: chunks={attempted}/{total} committed={committed} "
            f"validated={validated} ops_committed={ops_committed} "
            f"ops_validated={ops_validated}"
        )
    ]
    for issue in list(run.get("issues") or ())[:5]:
        if isinstance(issue, Mapping):
            lines.append(
                f"  [issue chunk={issue.get('chunk')}] "
                f"{issue.get('code')}: {issue.get('message')}"
            )
    return "\n".join(lines)


def _write_bulk_manifest(path: str | None, payload: Mapping[str, Any]) -> None:
    if not path:
        return
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")
    except OSError as exc:
        raise ValueError(f"cannot write bulk manifest {path!r}: {exc}") from exc
