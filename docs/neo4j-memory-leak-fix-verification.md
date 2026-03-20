# Neo4j Memory Leak Fix — Implementation Verification

Use this document **after** implementing fixes for the Neo4j driver/service leaks. Each section maps to a leak location and provides what to change and how to verify the fix.

---

## Fix status (last verified: 2025-03-02)

| # | Location | Status |
|---|----------|--------|
| 1 | ChangeDetectionTool | Fixed — `close()` added, `self.driver` → `self.neo4j_driver`, callers N/A (tool long-lived) |
| 2 | ParsingService | Fixed — `close()` added; controller, celery task, potpie/resources all use `finally` + `close()` |
| 3 | LibraryParsingService | Fixed — `close()` in parsing_adapter |
| 4 | change_detection inline InferenceService | Fixed — single instance, closed in `finally` |
| 5 | change_detection inline GetCodeFromNodeIdTool | Fixed — single instance, closed in `finally` |
| 6 | ask_knowledge_graph_queries_tool | Fixed — try/finally + `inference_service.close()` |
| 7 | knowledge_graph_router | Fixed — `finally` + `inference_service.close()` |
| 8 | search_semantic_tool | Fixed — try/finally + `inference_service.close()` |
| 9 | get_nodes_from_tags_tool | Fixed — `CodeGraphService` closed in `finally` |
| 10 | parsing_service cleanup path | Fixed — `code_graph_service.close()` in `finally` |
| 11 | parsing_adapter cleanup path | Fixed — both cleanup and _analyze_directory paths close `CodeGraphService` in `finally` |
| 12 | ToolService / shared driver | Optional — not implemented |

---

## How to test for leaks (local)

Use the script `scripts/neo4j_leak_test.py` to verify that connections and process memory stabilize (no leak).

**Prerequisites:** `.env` with `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, and Postgres (for direct mode). Neo4j must be running.

**Direct mode** — creates/closes a Neo4j driver in a loop (no app). Verifies that driver `close()` works and that RSS/connections stabilize. No running app needed:

```bash
# From repo root; 80 iterations, RSS sampling only
uv run python scripts/neo4j_leak_test.py --direct --iterations 80

# With Neo4j connection count (if Neo4j supports CALL dbms.listConnections() or SHOW CONNECTIONS)
uv run python scripts/neo4j_leak_test.py --direct --iterations 80 --neo4j-connections
```

**API mode (recommended for testing the fix)** — hits an endpoint so the app creates and closes Neo4j-backed services per request. Use `SERVER_PID` to sample backend memory.

- **semantic-search** — creates/closes `InferenceService` per request:
  ```bash
  BASE_URL=http://localhost:8001 AUTH_HEADER="Bearer <token>" PROJECT_ID=<your-project-uuid> \
    uv run python scripts/neo4j_leak_test.py --api --profile semantic-search --iterations 50
  ```
- **parse** — exercises `ParsingService` (and `CodeGraphService` in cleanup path); each request creates and closes them in the controller:
  ```bash
  BASE_URL=http://localhost:8001 AUTH_HEADER="Bearer <token>" \
    uv run python scripts/neo4j_leak_test.py --api --profile parse --iterations 30
  # Optional: REPO_NAME=owner/repo BRANCH_NAME=main
  ```

  Replace `12345` with the actual backend PID to sample server RSS:
  ```bash
  SERVER_PID=12345 BASE_URL=... AUTH_HEADER=... \
    uv run python scripts/neo4j_leak_test.py --api --profile parse --iterations 30
  ```

**Interpretation:** Script exits 0 if RSS growth (and Neo4j connection growth, if enabled) is within threshold; exits 1 if a leak is likely. See script docstring for thresholds.

---

## Verification Overview

| # | Location | Fix type | Recheck |
|---|----------|----------|---------|
| 1 | ChangeDetectionTool | Add close + fix bug | [§1](#1-change-detection-tool) |
| 2 | ParsingService | Close inference_service | [§2](#2-parsing-service) |
| 3 | LibraryParsingService | Close inference_service | [§3](#3-library-parsing-service) |
| 4 | change_detection inline InferenceService | Use context / close | [§4](#4-change-detection-inline-inference-service) |
| 5 | change_detection inline GetCodeFromNodeIdTool | Reuse or close | [§5](#5-change-detection-inline-getcodefromnodeidtool) |
| 6 | ask_knowledge_graph_queries_tool | Close InferenceService | [§6](#6-ask-knowledge-graph-queries-tool) |
| 7 | knowledge_graph_router | Close InferenceService | [§7](#7-knowledge-graph-router) |
| 8 | search_semantic_tool | Close InferenceService | [§8](#8-search-semantic-tool) |
| 9 | get_nodes_from_tags_tool | Close CodeGraphService | [§9](#9-get-nodes-from-tags-tool) |
| 10 | parsing_service cleanup path | Close CodeGraphService | [§10](#10-parsing-service-cleanup-path) |
| 11 | parsing_adapter cleanup path | Close CodeGraphService | [§11](#11-parsing-adapter-cleanup-path) |
| 12 | ToolService / shared driver (optional) | Centralize driver usage | [§12](#12-tool-service--shared-driver-optional) |

---

## 1. ChangeDetectionTool

**File:** `app/modules/intelligence/tools/change_detection/change_detection_tool.py`

**Required changes:**
- [ ] Add a `close()` method that calls `self.neo4j_driver.close()` (and set driver to `None` if desired).
- [ ] Fix bug: replace `self.driver` with `self.neo4j_driver` at line 332 (`traverse` method).
- [ ] Ensure every code path that creates a `ChangeDetectionTool` either calls `close()` when done or documents why the instance is long-lived and who closes it. If tools are created per-request and never closed, add a `close()` call at the appropriate lifecycle (e.g. request end) or use a shared driver (see §12).

**Recheck:**
```bash
# 1. close() exists and closes neo4j_driver
grep -n "def close\|neo4j_driver.close" app/modules/intelligence/tools/change_detection/change_detection_tool.py

# 2. No remaining self.driver (should be self.neo4j_driver)
grep -n "self\.driver" app/modules/intelligence/tools/change_detection/change_detection_tool.py
# Expected: no matches, or only in comments.
```

---

## 2. ParsingService

**File:** `app/modules/parsing/graph_construction/parsing_service.py`

**Required changes:**
- [ ] Add a `close()` method that calls `self.inference_service.close()`.
- [ ] Every caller that creates a `ParsingService` must call `close()` in a `finally` (or use a context manager):
  - `app/modules/parsing/graph_construction/parsing_controller.py` (parse_directory)
  - `app/celery/tasks/parsing_tasks.py` (process_parsing)
  - `potpie/resources/parsing.py` (parse_project, duplicate_graph if it creates ParsingService)

**Recheck:**
```bash
# 1. ParsingService has close()
grep -n "def close\|inference_service.close" app/modules/parsing/graph_construction/parsing_service.py

# 2. Callers use close/finally
grep -n "ParsingService(\|parsing_service\.close\|parsing_service)" app/modules/parsing/graph_construction/parsing_controller.py app/celery/tasks/parsing_tasks.py potpie/resources/parsing.py
# Manually confirm each ParsingService() is followed by try/finally with close() or equivalent.
```

---

## 3. LibraryParsingService

**File:** `potpie/services/parsing_adapter.py`

**Required changes:**
- [ ] Add a `close()` method that, if `self._inference_service` is not None, calls `self._inference_service.close()` and sets it to `None`.
- [ ] Any code that creates `LibraryParsingService` must call `close()` when the adapter is no longer needed, or document lifecycle and ensure a single long-lived adapter per process with explicit shutdown.

**Recheck:**
```bash
grep -n "def close\|_inference_service.close" potpie/services/parsing_adapter.py
```

---

## 4. ChangeDetectionTool — inline InferenceService

**File:** `app/modules/intelligence/tools/change_detection/change_detection_tool.py` (around 757–759)

**Required changes:**
- [ ] Use a single InferenceService for the method (e.g. create once at start of `get_code_changes` and close in `finally`), or use a context manager / try-finally and call `inference_service.close()` after use.

**Recheck:**
```bash
grep -n "InferenceService(" app/modules/intelligence/tools/change_detection/change_detection_tool.py
# For the inline use around 757: ensure it is inside try/finally with .close(), or replaced by a reused + closed instance.
```

---

## 5. ChangeDetectionTool — inline GetCodeFromNodeIdTool

**File:** `app/modules/intelligence/tools/change_detection/change_detection_tool.py` (lines 713, 731, 770)

**Required changes:**
- [ ] Create one `GetCodeFromNodeIdTool` instance at the start of the block (or reuse `self` if the tool is the same), use it for all calls, and in a `finally` call its `close()` if you add one—or ensure the single instance is closed when the method exits. Avoid creating a new tool (and thus driver) per loop iteration.

**Recheck:**
```bash
grep -n "GetCodeFromNodeIdTool(" app/modules/intelligence/tools/change_detection/change_detection_tool.py
# Should be one creation (or reuse) and no creation inside loops; if tool has close(), it should be in finally.
```

---

## 6. Ask Knowledge Graph Queries Tool

**File:** `app/modules/intelligence/tools/kg_based_tools/ask_knowledge_graph_queries_tool.py` (line 60)

**Required changes:**
- [ ] Wrap use of `inference_service` in try/finally and call `inference_service.close()` in `finally`.

**Recheck:**
```bash
grep -n "InferenceService(\|inference_service.close\|finally" app/modules/intelligence/tools/kg_based_tools/ask_knowledge_graph_queries_tool.py
# Should show InferenceService creation and a finally that closes it.
```

---

## 7. Knowledge Graph Router

**File:** `app/modules/knowledge_graph/knowledge_graph_router.py` (line 104)

**Required changes:**
- [ ] Add `finally` to the existing try block and call `inference_service.close()` in `finally`.

**Recheck:**
```bash
grep -n "inference_service\|finally\|close" app/modules/knowledge_graph/knowledge_graph_router.py
# Must have finally block that calls inference_service.close().
```

---

## 8. Search Semantic Tool

**File:** `app/modules/intelligence/tools/local_search_tools/search_semantic_tool.py` (around 77)

**Required changes:**
- [ ] After the block that uses `inference_service`, add try/finally and call `inference_service.close()` in `finally`.

**Recheck:**
```bash
grep -n "InferenceService(\|inference_service.close\|finally" app/modules/intelligence/tools/local_search_tools/search_semantic_tool.py
```

---

## 9. Get Nodes From Tags Tool

**File:** `app/modules/intelligence/tools/kg_based_tools/get_nodes_from_tags_tool.py` (lines 102–107)

**Required changes:**
- [ ] Assign `CodeGraphService(...)` to a variable, use it in try, and in `finally` call `code_graph_service.close()`.

**Recheck:**
```bash
grep -n "CodeGraphService(\|\.close()" app/modules/intelligence/tools/kg_based_tools/get_nodes_from_tags_tool.py
# CodeGraphService must be created, used, and closed in finally.
```

---

## 10. ParsingService Cleanup Path

**File:** `app/modules/parsing/graph_construction/parsing_service.py` (lines 207–215)

**Required changes:**
- [ ] After the `code_graph_service.cleanup_graph(...)` call, add `code_graph_service.close()` in a `finally` (or restructure so the same try/finally that creates CodeGraphService also closes it).

**Recheck:**
```bash
grep -n "code_graph_service\|cleanup_graph\|\.close()" app/modules/parsing/graph_construction/parsing_service.py
# In the cleanup_graph block (~207), ensure code_graph_service.close() is called (e.g. in finally).
```

---

## 11. Parsing Adapter Cleanup Path

**File:** `potpie/services/parsing_adapter.py` (lines 159–165)

**Required changes:**
- [ ] After `code_graph_service.cleanup_graph(project_id)`, call `code_graph_service.close()` in a `finally` block (or equivalent).

**Recheck:**
```bash
grep -n "code_graph_service\|cleanup_graph\|\.close()" potpie/services/parsing_adapter.py
# In the cleanup block (~159), ensure code_graph_service.close() is called.
```

---

## 12. ToolService / Shared Driver (Optional)

**Files:** `app/modules/intelligence/tools/tool_service.py`, `app/core/config_provider.py` (or a small neo4j module), and individual tools.

**Optional improvement:**
- [ ] Introduce a single shared sync Neo4j driver (or a thin wrapper) for the app, e.g. created once from config and passed into services/tools that need Neo4j. Tools and services then use this driver (or sessions from it) instead of creating their own. Shut down the shared driver at app shutdown.
- [ ] Ensure ToolService (or the layer that creates tools) does not create a new driver per tool instance; either pass the shared driver into tools or create tools once and reuse.

**Recheck:**
- If implemented: only one place creates the sync Neo4j driver for the app; all other code uses that instance or sessions from it; driver is closed on application shutdown.

---

## Post-Implementation Checklist

After all fixes:

1. **Code search**
   - [ ] No remaining `GraphDatabase.driver(` or `InferenceService(` or `CodeGraphService(` without a matching `close()` in the same scope or in a `finally`/context manager, except for:
     - The shared driver (if you introduced one),
     - Long-lived services that have a documented `close()` called at shutdown.
   - [ ] `ChangeDetectionTool` uses `self.neo4j_driver` everywhere (no `self.driver`).

2. **Tests**
   - [ ] Run existing tests: `pytest` (or project test command) and fix any regressions.
   - [ ] If possible, run a short load test (e.g. repeated parse or semantic search) and observe Neo4j connection count or client memory; both should stabilize instead of growing.

3. **Docs**
   - [ ] `docs/neo4j-memory-leak-analysis.md` — add a short “Fix status” section at the top with the date and “Fixed” for each item, or link to this verification doc.

---

## Quick Grep Summary (copy-paste)

Run from repo root to spot missing closes (manual review still required):

```bash
# All places that create a Neo4j driver or service
grep -rn "GraphDatabase\.driver\|InferenceService(\|CodeGraphService(" app potpie --include="*.py" | grep -v test

# All places that close drivers or services
grep -rn "\.close()\|inference_service\.close\|code_graph_service\.close\|neo4j_driver\.close" app potpie --include="*.py" | grep -v test
```

Compare the two lists: every creation in a short-lived scope should have a corresponding close in the same flow (same function or same request/task lifecycle).
