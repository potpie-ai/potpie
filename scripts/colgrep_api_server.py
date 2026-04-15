"""
Lightweight ColGREP API server for local/prod-like testing.

Mirrors the in-cluster ``colgrep-server`` contract:
  GET  /healthz
  POST /search  JSON: query, top_k, timeout_ms, optional target_paths

Legacy bodies with project_id / pattern / CLI flags are still accepted for local dev.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator


app = FastAPI(title="ColGREP API Server", version="0.2.0")


def _json_error_response(
    error: str,
    status_code: int,
    *,
    invalid_target_paths: Optional[List[str]] = None,
    latency_ms: Optional[int] = None,
) -> JSONResponse:
    body: Dict[str, Any] = {"error": error}
    if invalid_target_paths is not None:
        body["invalid_target_paths"] = invalid_target_paths
    if latency_ms is not None:
        body["latency_ms"] = latency_ms
    return JSONResponse(status_code=status_code, content=body)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    _request: Request, exc: RequestValidationError
) -> JSONResponse:
    errors = exc.errors()
    query_related = any("query" in [str(p) for p in err.get("loc", [])] for err in errors)
    query_required = any(
        "query is required" in str(err.get("msg", "")).lower()
        or "either query or pattern must be provided" in str(err.get("msg", "")).lower()
        for err in errors
    )
    if query_related or query_required:
        return _json_error_response("query is required", 400)
    return _json_error_response("invalid request", 400)


@app.exception_handler(HTTPException)
async def http_exception_handler(_request: Request, exc: HTTPException) -> JSONResponse:
    detail = exc.detail
    if isinstance(detail, dict) and "error" in detail:
        return JSONResponse(status_code=exc.status_code, content=detail)
    if isinstance(detail, str):
        return _json_error_response(detail, exc.status_code)
    return _json_error_response("request failed", exc.status_code)


class ColgrepSearchRequest(BaseModel):
    """Request body compatible with production colgrep-server and legacy shims."""

    query: Optional[str] = None
    top_k: int = Field(default=5, ge=1, le=100)
    timeout_ms: int = Field(default=120_000, ge=1_000, le=900_000)
    target_paths: Optional[List[str]] = None

    project_id: Optional[str] = Field(
        default=None,
        description="Legacy: select repo root via COLGREP_PROJECT_PATHS_JSON",
    )
    pattern: Optional[str] = None
    working_directory: Optional[str] = None
    conversation_id: Optional[str] = None
    include: List[str] = Field(default_factory=list)
    exclude: List[str] = Field(default_factory=list)
    exclude_dir: List[str] = Field(default_factory=list)
    files_only: bool = False
    content: bool = False
    code_only: bool = True
    context_lines: int = Field(default=6, ge=0, le=50)

    @model_validator(mode="after")
    def validate_query_or_pattern(self) -> "ColgrepSearchRequest":
        q = (self.query or "").strip()
        p = (self.pattern or "").strip()
        if not q and not p:
            raise ValueError("query is required")
        return self


def _resolve_colgrep_binary() -> str:
    configured = os.getenv("COLGREP_BINARY", "").strip()
    if configured and Path(configured).is_file():
        return configured
    local = Path(__file__).resolve().parents[1] / ".tools" / "bin" / "colgrep"
    if local.is_file():
        return str(local)
    return "colgrep"


def _resolve_root_path(project_id: Optional[str]) -> Path:
    key = project_id or "_default"
    project_map_raw = os.getenv("COLGREP_PROJECT_PATHS_JSON", "").strip()
    if project_map_raw:
        try:
            project_map = json.loads(project_map_raw)
            path = project_map.get(project_id) or project_map.get("_default")
            if path:
                resolved = Path(path).resolve()
                if resolved.is_dir():
                    return resolved
        except json.JSONDecodeError:
            pass
    default_path = os.getenv("COLGREP_SERVER_DEFAULT_PATH", "").strip()
    if default_path:
        resolved = Path(default_path).resolve()
        if resolved.is_dir():
            return resolved
    return Path.cwd().resolve()


def _validate_target_paths(
    root: Path, paths: Optional[List[str]]
) -> Optional[List[Path]]:
    if not paths:
        return None
    invalid: List[str] = []
    resolved_list: List[Path] = []
    for raw in paths:
        p = raw.strip()
        if not p:
            continue
        rel = Path(p)
        if rel.is_absolute() or ".." in rel.parts:
            invalid.append(raw)
            continue
        full = (root / rel).resolve()
        if not str(full).startswith(str(root)):
            invalid.append(raw)
            continue
        if not full.exists():
            invalid.append(raw)
            continue
        resolved_list.append(full)
    if invalid:
        raise HTTPException(
            status_code=400,
            detail={"error": "invalid target_paths", "invalid_target_paths": invalid},
        )
    return resolved_list or None


def _build_colgrep_command(req: ColgrepSearchRequest) -> List[str]:
    cmd: List[str] = [
        _resolve_colgrep_binary(),
        "search",
        "-y",
        "--json",
        "--results",
        str(req.top_k),
    ]
    if req.files_only:
        cmd.append("--files-only")
    if req.content:
        cmd.append("--content")
    elif req.context_lines:
        cmd.extend(["--lines", str(req.context_lines)])
    if req.code_only:
        cmd.append("--code-only")
    for g in req.include:
        cmd.extend(["--include", g])
    for g in req.exclude:
        cmd.extend(["--exclude", g])
    for d in req.exclude_dir:
        cmd.extend(["--exclude-dir", d])
    if req.pattern and req.pattern.strip():
        cmd.extend(["--pattern", req.pattern.strip()])
    if req.query and req.query.strip():
        cmd.append(req.query.strip())
    cmd.append(".")
    return cmd


def _parsed_to_response(
    parsed: Any,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Best-effort map of colgrep JSON stdout into API results / raw_results."""
    results: List[Dict[str, Any]] = []
    raw_results: List[Dict[str, Any]] = []

    def add_hit(obj: Dict[str, Any], score: Any = None) -> None:
        path = obj.get("path") or obj.get("file") or ""
        s = score if score is not None else obj.get("score")
        if path:
            results.append({"path": path, "score": s})
        unit = (
            obj.get("unit")
            if isinstance(obj.get("unit"), dict)
            else {
                "file": obj.get("file") or obj.get("path"),
                "line": obj.get("line"),
                "end_line": obj.get("end_line"),
                "language": obj.get("language"),
                "signature": obj.get("signature"),
                "code": obj.get("code"),
            }
        )
        raw_results.append({"unit": unit, "score": s})

    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict):
                sc = item.get("score")
                add_hit(item, sc)
    elif isinstance(parsed, dict):
        hits = parsed.get("results") or parsed.get("hits") or parsed.get("data")
        if isinstance(hits, list):
            return _parsed_to_response(hits)
        add_hit(parsed)

    return results, raw_results


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True}


@app.post("/search")
def search(req: ColgrepSearchRequest) -> dict:
    t0 = time.perf_counter()
    root = _resolve_root_path(req.project_id)

    validated_targets: Optional[List[Path]] = None
    merged_tp: Optional[List[str]] = None
    if req.target_paths:
        merged_tp = list(req.target_paths)
    elif req.working_directory and req.working_directory.strip():
        merged_tp = [req.working_directory.strip()]
    if merged_tp:
        validated_targets = _validate_target_paths(root, merged_tp)

    timeout_sec = min(
        req.timeout_ms / 1000.0,
        float(os.getenv("COLGREP_API_CMD_TIMEOUT_SEC", "600")),
    )

    env = os.environ.copy()
    xdg = os.getenv("COLGREP_XDG_DATA_HOME", "").strip()
    if xdg:
        env["XDG_DATA_HOME"] = xdg

    def run_once(cwd: Path) -> str:
        cmd = _build_colgrep_command(req)
        completed = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        stdout = (completed.stdout or "").strip()
        stderr = (completed.stderr or "").strip()
        if completed.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "colgrep failed",
                    "returncode": completed.returncode,
                    "stderr": stderr[:4000],
                    "stdout": stdout[:1000],
                },
            )
        return stdout

    def sort_key_score(r: Dict[str, Any]) -> float:
        s = r.get("score")
        try:
            return float(s)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return 0.0

    try:
        if validated_targets is not None and len(validated_targets) > 1:
            merged_results: List[Dict[str, Any]] = []
            merged_raw: List[Dict[str, Any]] = []
            for tp in validated_targets:
                stdout = run_once(tp)
                try:
                    parsed_merge: Any = json.loads(stdout)
                except json.JSONDecodeError:
                    latency_ms = int((time.perf_counter() - t0) * 1000)
                    return {
                        "results": [],
                        "raw_results": [],
                        "latency_ms": latency_ms,
                        "queue_wait_ms": 0,
                        "error": "colgrep output was not valid JSON",
                        "raw_stdout_preview": stdout[:800],
                    }
                r_part, rr_part = _parsed_to_response(parsed_merge)
                merged_results.extend(r_part)
                merged_raw.extend(rr_part)
            merged_results = sorted(
                merged_results, key=sort_key_score, reverse=True
            )[: req.top_k]
            latency_ms = int((time.perf_counter() - t0) * 1000)
            return {
                "results": merged_results,
                "raw_results": merged_raw[: max(50, req.top_k * 5)],
                "latency_ms": latency_ms,
                "queue_wait_ms": 0,
                "root_path": str(root),
            }

        if validated_targets is None:
            cwd = root
            stdout = run_once(cwd)
        else:
            stdout = run_once(validated_targets[0])
    except subprocess.TimeoutExpired:
        latency_ms = int(req.timeout_ms)
        return {
            "error": "colgrep search timed out",
            "latency_ms": latency_ms,
            "queue_wait_ms": 0,
            "results": [],
            "raw_results": [],
        }
    except HTTPException:
        raise
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"colgrep exec error: {e}") from e

    parsed: Any
    try:
        parsed = json.loads(stdout)
    except json.JSONDecodeError:
        parsed = None

    if parsed is None:
        latency_ms = int((time.perf_counter() - t0) * 1000)
        return {
            "results": [],
            "raw_results": [],
            "latency_ms": latency_ms,
            "queue_wait_ms": 0,
            "error": "colgrep output was not valid JSON",
            "raw_stdout_preview": stdout[:800],
        }

    results, raw_results = _parsed_to_response(parsed)

    if len(results) > req.top_k:
        results = sorted(results, key=sort_key_score, reverse=True)[: req.top_k]
    latency_ms = int((time.perf_counter() - t0) * 1000)
    return {
        "results": results,
        "raw_results": raw_results,
        "latency_ms": latency_ms,
        "queue_wait_ms": 0,
        "root_path": str(root),
    }
