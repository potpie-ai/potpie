"""
ColGREP Search Tool

Structured agent-facing access to ColGREP over HTTP API (production-like).
In production, ColGREP runs in a separate pod and is accessed via
``POST /search`` on ``colgrep-server`` (see service API guide).
"""

from __future__ import annotations

import asyncio
import os
import shlex
import time
from typing import Any, Dict, List, Optional

import httpx
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, model_validator
from sqlalchemy.orm import Session

from app.modules.intelligence.tools.code_query_tools.bash_command_tool import (
    BashCommandTool,
)
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)

_DEFAULT_API_TIMEOUT_MS = 120_000


class ColgrepSearchToolInput(BaseModel):
    project_id: str = Field(..., description="Project ID that references the repository")
    query: Optional[str] = Field(
        default=None,
        description="Natural language query for semantic code search.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Maximum number of ranked results to return (server top_k).",
    )
    timeout_ms: Optional[int] = Field(
        default=None,
        ge=1000,
        le=900_000,
        description="Server-side search budget in milliseconds (sent as timeout_ms). Defaults from COLGREP_API_TIMEOUT_MS.",
    )
    target_paths: Optional[List[str]] = Field(
        default=None,
        description="Optional repo-relative directory paths to narrow search (colgrep-server target_paths). Must not be absolute or escape the repo.",
    )
    working_directory: Optional[str] = Field(
        default=None,
        description="Optional subdirectory within the repository to search from; mapped to a single target_paths entry when target_paths is omitted.",
    )
    conversation_id: Optional[str] = Field(
        default=None,
        description="Optional conversation ID to search against the edits worktree (local fallback only).",
    )
    include: Optional[List[str]] = Field(
        default=None,
        description='Optional include globs, e.g. ["*.py", "*.ts"] (local CLI fallback only).',
    )
    exclude: Optional[List[str]] = Field(
        default=None,
        description='Optional exclude globs (local CLI fallback only).',
    )
    exclude_dir: Optional[List[str]] = Field(
        default=None,
        description='Optional excluded directories (local CLI fallback only).',
    )
    files_only: bool = Field(
        default=False,
        description="Return only matching file paths (local CLI fallback only).",
    )
    content: bool = Field(
        default=False,
        description="Return full function/class content (local CLI fallback only).",
    )
    code_only: bool = Field(
        default=True,
        description="Limit search to code files only (local CLI fallback only).",
    )
    context_lines: int = Field(
        default=6,
        ge=0,
        le=30,
        description="Context lines when not using content/files_only (local CLI fallback only).",
    )

    @model_validator(mode="after")
    def validate_query_or_pattern(self) -> "ColgrepSearchToolInput":
        if not (self.query and self.query.strip()):
            raise ValueError("query is required.")
        return self


def _default_timeout_ms() -> int:
    raw = os.getenv("COLGREP_API_TIMEOUT_MS", "").strip()
    if raw.isdigit() and int(raw) >= 1000:
        return int(raw)
    sec = os.getenv("COLGREP_API_TIMEOUT_SEC", "").strip()
    if sec:
        try:
            val = int(float(sec) * 1000)
            if val >= 1000:
                return val
        except ValueError:
            pass
    return _DEFAULT_API_TIMEOUT_MS


def _http_read_timeout_sec(timeout_ms: int) -> float:
    """Allow client read to outlive server budget slightly (network + queue)."""
    return max(timeout_ms / 1000.0 + 30.0, 60.0)


class ColgrepSearchTool:
    name = "search_colgrep"
    description = """Search a parsed repository using ColGREP semantic code search via the colgrep-server HTTP API.

    The service expects POST /search with JSON: query (required), top_k, timeout_ms, optional target_paths (repo-relative folders).
    Responses include results, raw_results (unit-level hits), latency_ms, and queue_wait_ms.

    Use this when you need indexed semantic search over a repository instead of plain text search.

    Best for:
    - finding code by meaning ("where is auth token validation handled?")
    - narrowing to a subsystem with target_paths or working_directory when you know the folder
    - inspecting ranked file paths and optional snippet-level raw_results

    Configuration:
    - COLGREP_API_BASE_URL (e.g. http://colgrep-server:8080)
    - COLGREP_API_TIMEOUT_MS for default timeout_ms (default 120000)
    - COLGREP_ALLOW_LOCAL_FALLBACK=1 to use local colgrep CLI via bash_command when API fails
    - Use check_colgrep_health to verify GET /healthz on the same base URL
    """
    args_schema: type[BaseModel] = ColgrepSearchToolInput

    def __init__(self, sql_db: Session, user_id: str):
        self.bash_tool = BashCommandTool(sql_db, user_id)
        self.api_base_url = os.getenv("COLGREP_API_BASE_URL", "").strip().rstrip("/")
        self.default_timeout_ms = _default_timeout_ms()
        self.allow_local_fallback = (
            os.getenv("COLGREP_ALLOW_LOCAL_FALLBACK", "").strip().lower()
            in {"1", "true", "yes", "on"}
        )

    @staticmethod
    def _quote(value: str) -> str:
        return shlex.quote(value)

    @staticmethod
    def _resolved_target_paths(
        input_data: ColgrepSearchToolInput,
    ) -> Optional[List[str]]:
        paths: List[str] = []
        if input_data.target_paths:
            paths.extend(
                [t.strip() for t in input_data.target_paths if t and str(t).strip()]
            )
        if input_data.working_directory and input_data.working_directory.strip():
            wd = input_data.working_directory.strip()
            if not input_data.target_paths:
                paths.append(wd)
        return paths or None

    def _build_command(self, input_data: ColgrepSearchToolInput) -> str:
        parts: list[str] = ["colgrep", "search", "-y", "--results", str(input_data.top_k)]

        if input_data.files_only:
            parts.append("--files-only")
        if input_data.content:
            parts.append("--content")
        elif input_data.context_lines:
            parts.extend(["--lines", str(input_data.context_lines)])
        if input_data.code_only:
            parts.append("--code-only")

        for pattern in input_data.include or []:
            parts.extend(["--include", pattern])
        for pattern in input_data.exclude or []:
            parts.extend(["--exclude", pattern])
        for directory in input_data.exclude_dir or []:
            parts.extend(["--exclude-dir", directory])
        if input_data.query and input_data.query.strip():
            parts.append(input_data.query.strip())

        parts.append(".")
        return " ".join(self._quote(part) for part in parts)

    def _build_api_payload(self, input_data: ColgrepSearchToolInput) -> Dict[str, Any]:
        timeout_ms = input_data.timeout_ms or self.default_timeout_ms
        payload: Dict[str, Any] = {
            "query": (input_data.query or "").strip(),
            "top_k": input_data.top_k,
            "timeout_ms": timeout_ms,
        }
        tps = self._resolved_target_paths(input_data)
        if tps:
            payload["target_paths"] = tps
        return payload

    @staticmethod
    def _format_api_error(
        body: Any, status_code: Optional[int] = None
    ) -> str:
        if isinstance(body, dict):
            err = body.get("error") or body.get("detail") or str(body)
            extra = []
            if "invalid_target_paths" in body:
                extra.append(
                    f"invalid_target_paths: {body['invalid_target_paths']}"
                )
            if "latency_ms" in body:
                extra.append(f"latency_ms={body['latency_ms']}")
            tail = "\n" + "\n".join(extra) if extra else ""
            prefix = f"HTTP {status_code}\n" if status_code else ""
            return f"❌ ColGREP API error.\n\n{prefix}{err}{tail}"
        return f"❌ ColGREP API error.\n\n{body!r}"

    @staticmethod
    def _format_api_success(body: Dict[str, Any]) -> str:
        if body.get("error"):
            return ColgrepSearchTool._format_api_error(body, None)

        lines: List[str] = []
        lm = body.get("latency_ms")
        qw = body.get("queue_wait_ms")
        if lm is not None or qw is not None:
            lines.append(
                f"(server timing: latency_ms={lm!s}, queue_wait_ms={qw!s})"
            )

        results = body.get("results") or []
        raw_results = body.get("raw_results") or []

        if not results and not raw_results:
            lines.append("📋 ColGREP search returned no results.")
            return "\n".join(lines)

        lines.append(f"Ranked paths ({len(results)}):")
        for i, r in enumerate(results[:25], 1):
            if isinstance(r, dict):
                path = r.get("path") or r.get("file") or str(r)
                score = r.get("score", "")
                lines.append(f"  {i}. {path}  score={score}")
            else:
                lines.append(f"  {i}. {r!r}")

        if raw_results:
            lines.append("")
            lines.append(f"Unit-level hits ({len(raw_results)}, showing up to 12):")
            for i, rr in enumerate(raw_results[:12], 1):
                if not isinstance(rr, dict):
                    lines.append(f"  {i}. {rr!r}")
                    continue
                unit = rr.get("unit") or {}
                score = rr.get("score", "")
                file = unit.get("file", "?")
                line = unit.get("line", "")
                lang = unit.get("language", "")
                sig = (unit.get("signature") or "")[:200]
                code = (unit.get("code") or "")[:600]
                lines.append(
                    f"  {i}. {file}:{line}  [{lang}]  score={score}"
                )
                if sig:
                    lines.append(f"      signature: {sig}")
                if code:
                    lines.append(f"      code:\n{code}")

        return "\n".join(lines) if lines else str(body)

    def _call_colgrep_api(self, input_data: ColgrepSearchToolInput) -> str:
        if not self.api_base_url:
            raise RuntimeError(
                "COLGREP_API_BASE_URL is not configured. Set it to the ColGREP service URL."
            )

        endpoint = f"{self.api_base_url}/search"
        payload = self._build_api_payload(input_data)
        timeout_ms = payload["timeout_ms"]
        read_sec = _http_read_timeout_sec(int(timeout_ms))

        logger.info(
            "[COLGREP_SEARCH] POST {} top_k={} timeout_ms={} target_paths={!r} query_len={}",
            endpoint,
            input_data.top_k,
            timeout_ms,
            payload.get("target_paths"),
            len(payload.get("query") or ""),
        )
        started = time.monotonic()
        with httpx.Client(
            timeout=httpx.Timeout(
                connect=15.0, read=read_sec, write=15.0, pool=15.0
            )
        ) as client:
            response = client.post(endpoint, json=payload)

        try:
            body = response.json()
        except Exception:
            body = {"error": (response.text or "")[:2000]}

        elapsed_ms = int((time.monotonic() - started) * 1000)
        logger.info(
            "[COLGREP_SEARCH] API status={} elapsed_ms={} project_id={} conversation_id={}",
            response.status_code,
            elapsed_ms,
            input_data.project_id,
            input_data.conversation_id,
        )

        if response.status_code >= 400:
            return self._format_api_error(body, response.status_code)

        if not isinstance(body, dict):
            return str(body)

        if body.get("error"):
            return self._format_api_error(body, response.status_code)

        return self._format_api_success(body)

    def _run_local_fallback(self, input_data: ColgrepSearchToolInput) -> str:
        command = self._build_command(input_data)
        # Prefer explicit worktree subdir; else first target_paths for cwd
        wd = input_data.working_directory
        if not wd and input_data.target_paths:
            wd = input_data.target_paths[0]

        result = self.bash_tool._run(
            project_id=input_data.project_id,
            command=command,
            working_directory=wd,
            conversation_id=input_data.conversation_id,
        )
        if not result.get("success"):
            error = result.get("error") or "Unknown ColGREP search error"
            return (
                "❌ ColGREP search failed.\n\n"
                f"Error: {error}\n\n"
                "Ensure the project has been parsed and the repository worktree is available."
            )

        output = (result.get("output") or "").strip()
        stderr = (result.get("error") or "").strip()
        if not output and not stderr:
            return "📋 ColGREP search returned no results."
        if output:
            return output
        return stderr

    def run(self, **kwargs: Any) -> str:
        input_data = ColgrepSearchToolInput(**kwargs)
        logger.info(
            "[COLGREP_SEARCH] Tool invoked project_id={} conversation_id={} working_directory={!r} target_paths={!r} query={!r} api_base_url={!r}",
            input_data.project_id,
            input_data.conversation_id,
            input_data.working_directory,
            input_data.target_paths,
            input_data.query,
            self.api_base_url,
        )
        try:
            return self._call_colgrep_api(input_data)
        except Exception as e:
            logger.warning("[COLGREP_SEARCH] API mode failed: {}", e)
            if self.allow_local_fallback:
                logger.info("[COLGREP_SEARCH] Falling back to local bash_command mode")
                return self._run_local_fallback(input_data)
            return (
                "❌ ColGREP API search failed.\n\n"
                f"Error: {e}\n\n"
                "This environment is configured for API-only ColGREP access. "
                "Ensure COLGREP_API_BASE_URL is reachable and the service accepts POST /search."
            )

    async def arun(self, **kwargs: Any) -> str:
        return await asyncio.to_thread(self.run, **kwargs)


class ColgrepHealthInput(BaseModel):
    """No fields; health check uses COLGREP_API_BASE_URL only."""

    model_config = {"extra": "ignore"}


class ColgrepHealthTool:
    name = "check_colgrep_health"
    description = """Check that the colgrep-server HTTP service is reachable (GET /healthz on COLGREP_API_BASE_URL).

    Use before relying on search_colgrep when debugging connectivity or deployment issues.
    Falls back to GET /health if /healthz returns 404 (older local shim).
    """
    args_schema: type[BaseModel] = ColgrepHealthInput

    def __init__(self, _sql_db: Session, _user_id: str):
        self.api_base_url = os.getenv("COLGREP_API_BASE_URL", "").strip().rstrip("/")

    def run(self, **kwargs: Any) -> str:
        _ = ColgrepHealthInput(**kwargs)
        if not self.api_base_url:
            return (
                "❌ COLGREP_API_BASE_URL is not set; cannot probe colgrep-server."
            )
        healthz = f"{self.api_base_url}/healthz"
        fallback = f"{self.api_base_url}/health"
        try:
            with httpx.Client(timeout=httpx.Timeout(10.0)) as client:
                r = client.get(healthz)
                if r.status_code == 404:
                    r = client.get(fallback)
                r.raise_for_status()
                try:
                    body = r.json()
                except Exception:
                    body = r.text
                return f"✅ ColGREP service OK ({r.url}): {body}"
        except Exception as e:
            return f"❌ ColGREP health check failed for {self.api_base_url}: {e}"

    async def arun(self, **kwargs: Any) -> str:
        return await asyncio.to_thread(self.run, **kwargs)


def colgrep_search_tool(sql_db: Session, user_id: str) -> Optional[StructuredTool]:
    tool_instance = ColgrepSearchTool(sql_db, user_id)
    if not tool_instance.api_base_url and not tool_instance.allow_local_fallback:
        logger.warning(
            "ColgrepSearchTool: COLGREP_API_BASE_URL not set and local fallback disabled; "
            "tool will return a configuration error at runtime"
        )
    return StructuredTool.from_function(
        coroutine=tool_instance.arun,
        func=tool_instance.run,
        name="search_colgrep",
        description=ColgrepSearchTool.description,
        args_schema=ColgrepSearchToolInput,
    )


def colgrep_health_tool(sql_db: Session, user_id: str) -> Optional[StructuredTool]:
    tool_instance = ColgrepHealthTool(sql_db, user_id)
    if not tool_instance.api_base_url:
        logger.warning(
            "ColgrepHealthTool: COLGREP_API_BASE_URL not set; health tool will report misconfiguration"
        )
    return StructuredTool.from_function(
        coroutine=tool_instance.arun,
        func=tool_instance.run,
        name="check_colgrep_health",
        description=ColgrepHealthTool.description,
        args_schema=ColgrepHealthInput,
    )
