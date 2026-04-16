"""
ColGREP Search Tool

Structured agent-facing access to ColGREP over HTTP API (production-like).
In production, ColGREP runs in a separate pod and is accessed via
``POST /search`` on ``colgrep-server`` (see service API guide).

Broad queries are automatically decomposed into keyword-heavy subqueries
and fanned out across individual target_paths to avoid timeouts on the
single-worker colgrep-server pod.
"""

from __future__ import annotations

import asyncio
import itertools
import os
import re
import shlex
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

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
_MAX_CONCURRENT_REQUESTS = 6
_BROAD_QUERY_WORD_THRESHOLD = 8
_BROAD_INTENT_PHRASES = re.compile(
    r"\b(?:how|implemented|implement|handling|handles|workflow|configuration|"
    r"configured|look\s+for|architecture|mechanism|responsible\s+for|"
    r"managed|manages|processing|processed|where\s+is|where\s+are|"
    r"what\s+is|what\s+are|explain|overview|describe)\b",
    re.IGNORECASE,
)
_MULTI_CONCEPT_SEP = re.compile(r",\s*(?:and\s+)?|;\s*|\s+and\s+")


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


# ---------------------------------------------------------------------------
# Timeout / config helpers
# ---------------------------------------------------------------------------

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
    return max(timeout_ms / 1000.0 + 30.0, 60.0)


# ---------------------------------------------------------------------------
# Search plan data structures
# ---------------------------------------------------------------------------

@dataclass
class SearchTask:
    query: str
    target_path: Optional[str]
    top_k: int
    timeout_ms: int


@dataclass
class SearchTaskResult:
    task: SearchTask
    results: List[Dict[str, Any]] = field(default_factory=list)
    raw_results: List[Dict[str, Any]] = field(default_factory=list)
    latency_ms: int = 0
    queue_wait_ms: int = 0
    error: Optional[str] = None
    timed_out: bool = False


@dataclass
class MergedSearchOutput:
    results: List[Dict[str, Any]]
    raw_results: List[Dict[str, Any]]
    subqueries: List[str]
    target_paths_searched: List[str]
    total_tasks: int
    succeeded: int
    failed: int
    timed_out: int
    is_partial: bool
    total_latency_ms: int


# ---------------------------------------------------------------------------
# Broad-query detection & decomposition
# ---------------------------------------------------------------------------

def _is_broad_query(query: str) -> bool:
    q = query.strip()
    words = q.split()
    if len(words) >= _BROAD_QUERY_WORD_THRESHOLD:
        return True
    if _BROAD_INTENT_PHRASES.search(q):
        return True
    segments = _MULTI_CONCEPT_SEP.split(q)
    non_trivial = [s.strip() for s in segments if len(s.strip().split()) >= 2]
    if len(non_trivial) >= 2:
        return True
    return False


def _decompose_query(query: str) -> List[str]:
    q = query.strip()

    segments = _MULTI_CONCEPT_SEP.split(q)
    non_trivial = [s.strip() for s in segments if s.strip()]

    if len(non_trivial) >= 2:
        subqueries = []
        for seg in non_trivial[:4]:
            kw = _extract_keywords(seg)
            subqueries.append(kw if kw else seg)
        return subqueries

    kw_full = _extract_keywords(q)
    words = kw_full.split() if kw_full else q.split()

    if len(words) <= 4:
        if not _is_broad_query(q):
            return [" ".join(words)]
        if len(words) <= 1:
            return [" ".join(words)]
        midpoint = max(1, len(words) // 2)
        chunks = [
            " ".join(words[:midpoint]).strip(),
            " ".join(words[midpoint:]).strip(),
        ]
        return [chunk for chunk in chunks if chunk]

    chunk_size = max(3, len(words) // 3)
    chunks: List[str] = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks[:4]


_STOPWORDS = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would shall should may might can could of in on at to for "
    "with by from as into through during before after above below "
    "between out off over under again further then once here there "
    "when where why how all each every both few more most other some "
    "such no nor not only own same so than too very it its that this "
    "what which who whom and or but if".split()
)


def _extract_keywords(text: str) -> str:
    words = re.findall(r"[A-Za-z0-9_]+", text)
    kept = [w for w in words if w.lower() not in _STOPWORDS and len(w) > 1]
    return " ".join(kept)


# ---------------------------------------------------------------------------
# ColgrepSearchTool
# ---------------------------------------------------------------------------

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

    Broad natural-language queries are automatically decomposed into smaller keyword-focused
    subqueries and fanned out across individual target_paths for reliability.

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

    # ------------------------------------------------------------------
    # Helpers preserved from previous implementation
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Search plan construction
    # ------------------------------------------------------------------

    @staticmethod
    def _needs_decomposition(query: str, target_paths: Optional[List[str]]) -> bool:
        multi_paths = target_paths is not None and len(target_paths) > 1
        broad = _is_broad_query(query)
        return broad or multi_paths

    @staticmethod
    def _build_search_plan(
        query: str,
        target_paths: Optional[List[str]],
        top_k: int,
        timeout_ms: int,
    ) -> List[SearchTask]:
        broad = _is_broad_query(query)
        subqueries = _decompose_query(query) if broad else [query.strip()]
        path_list: List[Optional[str]] = (
            [p for p in target_paths] if target_paths else [None]
        )

        per_task_top_k = max(top_k, 10)

        tasks: List[SearchTask] = []
        for sq, tp in itertools.product(subqueries, path_list):
            tasks.append(SearchTask(
                query=sq,
                target_path=tp,
                top_k=per_task_top_k,
                timeout_ms=timeout_ms,
            ))
        return tasks

    # ------------------------------------------------------------------
    # Single-request API call (sync, used for narrow fast path)
    # ------------------------------------------------------------------

    def _call_colgrep_api_single_sync(
        self,
        query: str,
        top_k: int,
        timeout_ms: int,
        target_path: Optional[str],
    ) -> Dict[str, Any]:
        endpoint = f"{self.api_base_url}/search"
        payload: Dict[str, Any] = {
            "query": query,
            "top_k": top_k,
            "timeout_ms": timeout_ms,
        }
        if target_path:
            payload["target_paths"] = [target_path]

        read_sec = _http_read_timeout_sec(timeout_ms)
        with httpx.Client(
            timeout=httpx.Timeout(connect=15.0, read=read_sec, write=15.0, pool=15.0)
        ) as client:
            response = client.post(endpoint, json=payload)

        try:
            body = response.json()
        except Exception:
            body = {"error": (response.text or "")[:2000]}

        if response.status_code >= 400:
            if isinstance(body, dict):
                body.setdefault("error", f"HTTP {response.status_code}")
            else:
                body = {"error": f"HTTP {response.status_code}: {body!r}"}
        return body if isinstance(body, dict) else {"error": str(body)}

    # ------------------------------------------------------------------
    # Async single-request (used by fanout path)
    # ------------------------------------------------------------------

    async def _call_colgrep_api_async(
        self, task: SearchTask
    ) -> SearchTaskResult:
        endpoint = f"{self.api_base_url}/search"
        payload: Dict[str, Any] = {
            "query": task.query,
            "top_k": task.top_k,
            "timeout_ms": task.timeout_ms,
        }
        if task.target_path:
            payload["target_paths"] = [task.target_path]

        read_sec = _http_read_timeout_sec(task.timeout_ms)
        started = time.monotonic()
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=15.0, read=read_sec, write=15.0, pool=15.0
                )
            ) as client:
                response = await client.post(endpoint, json=payload)

            try:
                body = response.json()
            except Exception:
                body = {"error": (response.text or "")[:2000]}

            elapsed = int((time.monotonic() - started) * 1000)

            if not isinstance(body, dict):
                return SearchTaskResult(
                    task=task, error=str(body), latency_ms=elapsed
                )

            if response.status_code >= 400 or body.get("error"):
                err = body.get("error") or f"HTTP {response.status_code}"
                is_timeout = "timed out" in str(err).lower()
                return SearchTaskResult(
                    task=task,
                    error=err,
                    timed_out=is_timeout,
                    latency_ms=body.get("latency_ms", elapsed),
                )

            return SearchTaskResult(
                task=task,
                results=body.get("results") or [],
                raw_results=body.get("raw_results") or [],
                latency_ms=body.get("latency_ms", elapsed),
                queue_wait_ms=body.get("queue_wait_ms", 0),
            )
        except Exception as exc:
            elapsed = int((time.monotonic() - started) * 1000)
            is_timeout = "timeout" in str(exc).lower() or "timed out" in str(exc).lower()
            return SearchTaskResult(
                task=task,
                error=str(exc),
                timed_out=is_timeout,
                latency_ms=elapsed,
            )

    # ------------------------------------------------------------------
    # Fanout execution with bounded concurrency
    # ------------------------------------------------------------------

    async def _execute_search_plan(
        self, tasks: List[SearchTask]
    ) -> List[SearchTaskResult]:
        semaphore = asyncio.Semaphore(_MAX_CONCURRENT_REQUESTS)

        async def limited(t: SearchTask) -> SearchTaskResult:
            async with semaphore:
                return await self._call_colgrep_api_async(t)

        return list(await asyncio.gather(*(limited(t) for t in tasks)))

    # ------------------------------------------------------------------
    # Result merging
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_search_results(
        task_results: List[SearchTaskResult],
        final_top_k: int,
    ) -> MergedSearchOutput:
        subqueries_used = sorted(
            {tr.task.query for tr in task_results}
        )
        paths_searched = sorted(
            {tr.task.target_path for tr in task_results if tr.task.target_path}
        )

        succeeded = sum(1 for tr in task_results if tr.error is None)
        failed = sum(1 for tr in task_results if tr.error and not tr.timed_out)
        timed_out = sum(1 for tr in task_results if tr.timed_out)
        total_latency = max((tr.latency_ms for tr in task_results), default=0)

        path_best_score: Dict[str, float] = {}
        path_subquery_hits: Dict[str, set[str]] = {}
        raw_by_path: Dict[str, List[Dict[str, Any]]] = {}

        for tr in task_results:
            if tr.error:
                continue
            for r in tr.results:
                p = r.get("path") or r.get("file") or ""
                if not p:
                    continue
                try:
                    s = float(r.get("score") or 0)
                except (TypeError, ValueError):
                    s = 0.0
                if p not in path_best_score or s > path_best_score[p]:
                    path_best_score[p] = s
                path_subquery_hits.setdefault(p, set()).add(tr.task.query)
            for rr in tr.raw_results:
                unit = rr.get("unit") or {}
                p = unit.get("file") or ""
                if p:
                    raw_by_path.setdefault(p, []).append(rr)

        def sort_key(path: str) -> Tuple[int, float, str]:
            return (
                -len(path_subquery_hits.get(path, set())),
                -path_best_score.get(path, 0.0),
                path,
            )

        ranked_paths = sorted(path_best_score.keys(), key=sort_key)[:final_top_k]

        results: List[Dict[str, Any]] = []
        for p in ranked_paths:
            results.append({
                "path": p,
                "score": path_best_score[p],
                "hit_count": len(path_subquery_hits.get(p, set())) or 1,
            })

        raw_results: List[Dict[str, Any]] = []
        seen_raw: set = set()
        for p in ranked_paths:
            for rr in raw_by_path.get(p, []):
                unit = rr.get("unit") or {}
                key = (unit.get("file", ""), unit.get("line", ""), unit.get("signature", ""))
                if key not in seen_raw:
                    seen_raw.add(key)
                    raw_results.append(rr)

        return MergedSearchOutput(
            results=results,
            raw_results=raw_results[:50],
            subqueries=subqueries_used,
            target_paths_searched=paths_searched,
            total_tasks=len(task_results),
            succeeded=succeeded,
            failed=failed,
            timed_out=timed_out,
            is_partial=(failed + timed_out > 0 and succeeded > 0),
            total_latency_ms=total_latency,
        )

    # ------------------------------------------------------------------
    # Output formatting
    # ------------------------------------------------------------------

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

    @staticmethod
    def _format_merged_output(merged: MergedSearchOutput) -> str:
        lines: List[str] = []

        meta_parts = [
            f"subqueries={len(merged.subqueries)}",
            f"paths={len(merged.target_paths_searched) or 'repo-wide'}",
            f"requests={merged.total_tasks}",
            f"ok={merged.succeeded}",
        ]
        if merged.timed_out:
            meta_parts.append(f"timed_out={merged.timed_out}")
        if merged.failed:
            meta_parts.append(f"failed={merged.failed}")
        meta_parts.append(f"latency_ms={merged.total_latency_ms}")
        lines.append(f"(search plan: {', '.join(meta_parts)})")

        if merged.is_partial:
            lines.append("⚠️ Partial results — some requests timed out or failed.")

        if merged.subqueries and len(merged.subqueries) > 1:
            lines.append(f"Subqueries: {merged.subqueries}")
        if merged.target_paths_searched:
            lines.append(f"Paths searched: {merged.target_paths_searched}")

        if not merged.results:
            lines.append("📋 ColGREP search returned no results.")
            return "\n".join(lines)

        lines.append(f"\nRanked paths ({len(merged.results)}):")
        for i, r in enumerate(merged.results[:25], 1):
            path = r.get("path", "?")
            score = r.get("score", "")
            hits = r.get("hit_count", 1)
            extra = f"  (appeared in {hits} subqueries)" if hits > 1 else ""
            lines.append(f"  {i}. {path}  score={score}{extra}")

        if merged.raw_results:
            lines.append("")
            lines.append(
                f"Unit-level hits ({len(merged.raw_results)}, showing up to 12):"
            )
            for i, rr in enumerate(merged.raw_results[:12], 1):
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
                lines.append(f"  {i}. {file}:{line}  [{lang}]  score={score}")
                if sig:
                    lines.append(f"      signature: {sig}")
                if code:
                    lines.append(f"      code:\n{code}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Original single-request path (narrow fast path)
    # ------------------------------------------------------------------

    def _call_colgrep_api(self, input_data: ColgrepSearchToolInput) -> str:
        if not self.api_base_url:
            raise RuntimeError(
                "COLGREP_API_BASE_URL is not configured. Set it to the ColGREP service URL."
            )

        query = (input_data.query or "").strip()
        timeout_ms = input_data.timeout_ms or self.default_timeout_ms
        tps = self._resolved_target_paths(input_data)

        if self._needs_decomposition(query, tps):
            return self._call_colgrep_api_decomposed(
                query=query,
                target_paths=tps,
                top_k=input_data.top_k,
                timeout_ms=timeout_ms,
            )

        target_path = tps[0] if tps and len(tps) == 1 else None

        logger.info(
            "[COLGREP_SEARCH] narrow fast path: query_len={} target_path={!r}",
            len(query),
            target_path,
        )
        started = time.monotonic()
        body = self._call_colgrep_api_single_sync(
            query=query,
            top_k=input_data.top_k,
            timeout_ms=timeout_ms,
            target_path=target_path,
        )
        elapsed_ms = int((time.monotonic() - started) * 1000)

        logger.info(
            "[COLGREP_SEARCH] narrow fast path done elapsed_ms={} project_id={}",
            elapsed_ms,
            input_data.project_id,
        )

        if body.get("error"):
            return self._format_api_error(body, None)
        return self._format_api_success(body)

    # ------------------------------------------------------------------
    # Decomposed / fanout path
    # ------------------------------------------------------------------

    def _call_colgrep_api_decomposed(
        self,
        query: str,
        target_paths: Optional[List[str]],
        top_k: int,
        timeout_ms: int,
    ) -> str:
        tasks = self._build_search_plan(query, target_paths, top_k, timeout_ms)
        logger.info(
            "[COLGREP_SEARCH] decomposed path: {} tasks from query_len={} paths={}",
            len(tasks),
            len(query),
            len(target_paths) if target_paths else 0,
        )

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                task_results = pool.submit(
                    lambda: asyncio.run(self._execute_search_plan(tasks))
                ).result()
        else:
            task_results = asyncio.run(self._execute_search_plan(tasks))

        merged = self._merge_search_results(task_results, top_k)

        if merged.succeeded == 0:
            errors = [tr.error for tr in task_results if tr.error]
            sample = "; ".join(errors[:3])
            return (
                f"❌ All {merged.total_tasks} ColGREP search requests failed.\n\n"
                f"Errors: {sample}\n\n"
                f"Subqueries attempted: {merged.subqueries}\n"
                f"Paths attempted: {merged.target_paths_searched or ['(repo-wide)']}"
            )

        return self._format_merged_output(merged)

    # ------------------------------------------------------------------
    # Local fallback (unchanged)
    # ------------------------------------------------------------------

    def _run_local_fallback(self, input_data: ColgrepSearchToolInput) -> str:
        command = self._build_command(input_data)
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

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Health check tool (unchanged)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Factory functions (unchanged)
# ---------------------------------------------------------------------------

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
