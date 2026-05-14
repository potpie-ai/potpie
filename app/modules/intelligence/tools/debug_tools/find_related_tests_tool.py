"""Find test files related to a given source file or symbol.

Strategy (in order, stops at first useful result):
1. Knowledge-graph query for callers of the symbol that live in test files.
2. ripgrep via sandbox_search for import of the file's basename.
3. Path-mirror heuristic: src/foo/bar.ts → tests/foo/bar.test.ts, etc.
"""

from __future__ import annotations

import os
import re
from typing import List, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)

_TEST_DIR_CANDIDATES = ["tests", "test", "__tests__", "spec", "e2e"]
_TEST_EXT_MAP = {
    ".py": [".test.py", "_test.py"],
    ".ts": [".test.ts", ".spec.ts"],
    ".js": [".test.js", ".spec.js"],
    ".go": ["_test.go"],
    ".java": ["Test.java"],
    ".rb": ["_spec.rb"],
}


def _basename_without_ext(file_path: str) -> str:
    base = os.path.basename(file_path)
    name, _ = os.path.splitext(base)
    return name


def _path_mirror_candidates(file_path: str) -> List[str]:
    """Generate mirror test path candidates for a source file."""
    name, ext = os.path.splitext(os.path.basename(file_path))
    suffixes = _TEST_EXT_MAP.get(ext, [f".test{ext}", f".spec{ext}", f"_test{ext}"])
    parts = re.split(r"[/\\]", file_path)

    candidates = []
    for test_dir in _TEST_DIR_CANDIDATES:
        # Replace first path component that looks like src/lib/app with test_dir
        for i, part in enumerate(parts):
            if part in ("src", "lib", "app", "source", "main"):
                mirror_parts = parts[:i] + [test_dir] + parts[i + 1 :]
                base_path = "/".join(mirror_parts[:-1])
                for suffix in suffixes:
                    candidates.append(f"{base_path}/{name}{suffix}")
                break
        # Also try flat: <test_dir>/<filename><suffix>
        for suffix in suffixes:
            candidates.append(f"{test_dir}/{name}{suffix}")

    return candidates


def find_related_tests(input_data: "FindRelatedTestsInput") -> str:
    file_path = input_data.file_path or ""
    symbol = input_data.symbol or ""

    results: List[str] = []
    methods_used: List[str] = []

    # --- Strategy 1: knowledge graph ---
    if symbol:
        try:
            from app.modules.intelligence.tools.code_tools.ask_knowledge_graph_queries_tool import (
                AskKnowledgeGraphQueriesInput,
                ask_knowledge_graph_queries_tool,
            )

            query = f"Which test files or test functions call or import '{symbol}'?"
            kg_result = ask_knowledge_graph_queries_tool(
                AskKnowledgeGraphQueriesInput(query=query)
            )
            if kg_result and "test" in kg_result.lower():
                results.append(f"**Knowledge graph results for `{symbol}`:**\n{kg_result}")
                methods_used.append("knowledge_graph")
        except Exception as e:
            logger.debug(f"find_related_tests: KG strategy failed: {e}")

    # --- Strategy 2: ripgrep via sandbox_search ---
    if file_path:
        basename = _basename_without_ext(file_path)
        try:
            from app.modules.intelligence.tools.local_search_tools.search_text_tool import (
                SearchTextInput,
                search_text_tool,
            )

            rg_result = search_text_tool(
                SearchTextInput(
                    query=basename,
                    file_pattern="**/*{test,spec}*",
                    case_sensitive=False,
                    use_regex=False,
                    max_results=20,
                )
            )
            if rg_result and "❌" not in rg_result:
                results.append(f"**ripgrep search for `{basename}` in test files:**\n{rg_result}")
                methods_used.append("ripgrep")
        except Exception as e:
            logger.debug(f"find_related_tests: ripgrep strategy failed: {e}")

    # --- Strategy 3: path-mirror heuristic ---
    if file_path:
        candidates = _path_mirror_candidates(file_path)
        mirror_hits = candidates[:8]  # surface top candidates to the agent
        results.append(
            "**Path-mirror candidates** (check if these exist):\n"
            + "\n".join(f"  - `{c}`" for c in mirror_hits)
        )
        methods_used.append("path_mirror")

    if not results:
        return f"No related tests found for file=`{file_path}` symbol=`{symbol}`. Try searching manually with `sandbox_search`."

    header = f"Related tests for {'`' + file_path + '`' if file_path else ''}{' / `' + symbol + '`' if symbol else ''} (strategies: {', '.join(methods_used)}):\n"
    return header + "\n\n".join(results)


class FindRelatedTestsInput(BaseModel):
    file_path: Optional[str] = Field(
        default=None,
        description="Source file path (e.g. src/checkout/createOrder.ts). Used for ripgrep and path-mirror strategies.",
    )
    symbol: Optional[str] = Field(
        default=None,
        description="Function or class name (e.g. createOrder). Used for knowledge-graph strategy.",
    )


def create_find_related_tests_tool() -> StructuredTool:
    return StructuredTool.from_function(
        func=find_related_tests,
        name="find_related_tests",
        description=(
            "Find test files related to a source file or symbol using three strategies: "
            "knowledge-graph callers, ripgrep import search, and path-mirror heuristics. "
            "Provide file_path, symbol, or both."
        ),
        args_schema=FindRelatedTestsInput,
    )
