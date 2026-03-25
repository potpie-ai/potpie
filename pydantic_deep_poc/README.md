# pydantic-deep PoC (vs Potpie `CodeGenAgent`)

Isolated project under `potpie/pydantic_deep_poc/` using [pydantic-deep](https://github.com/vstorm-co/pydantic-deepagents) with **OpenRouter** and scoped toolsets that mirror Potpie’s supervisor / execute / integration allow-lists (stubs for Jira/Confluence/Linear; real git + subprocess + in-memory Code Changes Manager).

## Setup

1. **Python**: 3.10+ recommended; use the PoC venv only (do not mix with Potpie’s `pydantic-ai` install).

   ```bash
   cd pydantic_deep_poc
   uv venv .venv
   source .venv/bin/activate
   uv pip install -e .
   ```

2. **Environment**: copy `.env.poc.example` to `.env.poc` and set `OPENROUTER_API_KEY` (and optional `MODEL_NAME`, default `moonshotai/kimi-k2.5`). For Logfire, set `LOGFIRE_TOKEN` (optional; tracing still initializes locally without cloud send).

3. **Test repo**: bare-clone `https://github.com/potpie-ai/potpie` into `pydantic_deep_poc/.repos/potpie.git` and add a base worktree at `.repos/potpie/main` (tries `main`, then `master`):

   ```bash
   PYTHONPATH=. python scripts/setup_repo.py
   ```

   Or: `PYTHONPATH=. python -m poc.repo_setup`. Per-run branches use `create_worktree` via `init_run_context()`.

## OpenRouter smoke test

```bash
set -a && source .env.poc && set +a
PYTHONPATH=. python -m poc.config.smoke_openrouter
```

## Logfire

Benchmarks call `initialize_logfire_tracing()` once per process (via `poc.benchmarks.tracing_setup`), wrap each agent run with `logfire_trace_metadata(scenario=..., run_id=..., model=..., impl=...)`, and `shutdown_logfire_tracing()` in a `finally` block. With `LOGFIRE_TOKEN` set and `LOGFIRE_SEND_TO_CLOUD=true`, spans and baggage show up in the Logfire UI.

## Benchmarks

Each script expects `OPENROUTER_API_KEY` and `PYTHONPATH=.` from `pydantic_deep_poc` root.

| Script | Description |
|--------|-------------|
| `python -m poc.benchmarks.scenario_1_single` | Single agent, THINK_EXECUTE tools only, no subagents |
| `python -m poc.benchmarks.scenario_2_multi` | Orchestrator with `discover` / `implement` / `verify` workers |
| `python -m poc.benchmarks.scenario_3_streaming` | Same timing as multi; TTFT via `run_stream` left to pydantic-ai docs |
| `python -m poc.benchmarks.scenario_4_quality` | Read-heavy quality task |
| `python -m poc.benchmarks.scenario_5_async` | Prompt mentions async `task` / `check_task` |

Override the task with `POC_TASK=...`.

## Comparison harness

```bash
set -a && source .env.poc && set +a
PYTHONPATH=. python -m poc.comparison.harness
```

Writes `poc/comparison/results/YYYY-MM-DD.md` with a deep-agent sample. Potpie `CodeGenAgent` must be run separately in the **main** Potpie venv with a wired `ChatContext` (DI, DB, etc.); this PoC does not instantiate that stack.

## Architecture

- **`PoCDeepDeps`**: extends `DeepAgentDeps` with shared `poc_run: RunContext` (CCM + todos + requirements + worktree path) across supervisor and subagents via `clone_for_subagent`.
- **`create_deep_agent`**: `include_todo=False`, `include_filesystem=False`, `include_plan=False`, `include_general_purpose_subagent=False`, `cost_tracking=False`, `context_manager=False`, `deps_type=PoCDeepDeps`. The PoC now disables pydantic-deep’s built-in subagent injection and adds a custom `create_subagent_toolset(...)` so workers only get their intended toolsets.
- **Prompts**: `code_gen_task_prompt` is loaded from the Potpie repo when present (`../app/.../code_gen_agent.py`); otherwise a short fallback string is used.
- **Skills**: local skills under `pydantic_deep_poc/skills/` are enabled for reusable migration playbooks (`large-code-migration`, `celery-to-hatchet`).

## Gap analysis (production parity)

| Area | PoC | Potpie |
|------|-----|--------|
| Delegation UX | `task` / `check_task` (subagents-pydantic-ai) | `delegate_to_*_agent` tools |
| Tool registry | Static `FunctionToolset` lists in `toolsets_builder.py` | `ToolResolver` + `definitions.py` allow-lists |
| CCM | In-memory `RunContext.code_changes` | Redis + repo manager |
| KG tools | Omitted | `get_code_from_*`, `ask_knowledge_graph_queries`, … |
| Tunnel / local mode | Omitted | `init_managers` + extension routing |
| MCP servers | Omitted | `AgentFactory.create_mcp_servers` |
| Embeddings / inferring | Omitted | `exclude_embedding_tools` |
| Auth / DI | Omitted | FastAPI + services |

## Dependency isolation

This project uses **`pydantic-ai-slim`** via `pydantic-deep`. Do not install it into Potpie’s primary environment without resolving version conflicts with `pydantic-ai>=1.56`.
