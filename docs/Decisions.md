# Experience Engine — Senior Engineer Review & Integration Plan for Potpie

> Review by: Senior ML Engineer  
> Date: March 2026  
> Context: Reviewing the "Experience Engine" proposal for integration into [potpie-ai/potpie](https://github.com/potpie-ai/potpie)

---

## 1. Overall Assessment

The document is well-structured and the core thesis is correct: coding agents need cross-session memory, existing tools don't solve it, and the two-path retrieval (file-anchored + semantic) with a scope hierarchy is a genuinely strong architecture. The junior's self-critique of the PoC is honest and mostly accurate. The v2 design fixes real problems.

That said, I'm going to be direct about where the design is overengineered for a first release, where assumptions are untested, and where the integration with potpie's actual architecture has gaps that need answering before you write code.

---

## 2. What's Strong — Ship These Ideas

### 2.1 File-Anchored Retrieval (Path A)

This is the single best idea in the document. Semantic search alone has a recall problem that nobody in the "memory for agents" space has solved well — you're always one paraphrase away from missing the lesson. Anchoring to file paths and using a deterministic SQLite lookup is the correct answer for the highest-value retrieval case: "you're about to edit the same file where the last disaster happened."

**Potpie-specific advantage:** Potpie already has a Neo4j knowledge graph with every file, function, and class indexed. You have the file paths. You have the relationships. Path A should be trivial to populate.

### 2.2 The Scope Hierarchy Concept

`file → project → developer → org` is the right abstraction. The Docker OOM example in the doc is convincing. Lessons should travel.

### 2.3 Async Post-Mortem Extraction

Don't block the user. Extract lessons after the session. Correct. Potpie already uses Celery + Redis for background work (repo parsing, knowledge graph construction). The infrastructure exists.

### 2.4 Per-Project Invalidation

This is an underrated design choice. The "lesson is wrong in repo X but correct in repos A through W" scenario is real in microservice architectures. The `invalidated_in_projects` dict is elegant.

---

## 3. What's Overengineered — Simplify for v1

### 3.1 The Four-Tier Scope System Is a v2 Feature

You've defined FILE, PROJECT, DEVELOPER, ORG — four scopes, each with promotion thresholds, confidence floors, and separate retrieval queries. For a first release, this is too many moving parts to debug simultaneously.

**My recommendation:** Ship with two scopes — `PROJECT` and `DEVELOPER`. Here's why:

- **FILE scope is just PROJECT with a file anchor.** You don't need a separate scope enum value. A project-scoped lesson with `file_anchors=["docker/Dockerfile"]` already fires on that file via Path A. The "scope" distinction adds schema complexity for zero retrieval benefit.
- **ORG scope requires multi-tenant infrastructure you don't have yet.** The doc says "never auto-promote to org, requires human approval." That means you need an approval UI, org-level admin roles, a review queue. That's a product feature, not a memory engine feature. Park it.

**Action:** Start with `project` and `developer` scopes. A lesson with file anchors gets retrieved via Path A within its project. A developer-scoped lesson fires across repos via tech-stack overlap in Path B. That covers 90% of the value. Add FILE and ORG scopes when you have usage data showing they're needed.

### 3.2 The Promotion Engine Is Premature

The automatic promotion (`evaluate_promotion` running after every reinforcement) is the kind of thing that sounds elegant in a design doc and creates bizarre edge cases in production:

- What counts as "reinforced"? The doc says `times_reinforced` increments, but when? When the lesson fires and the user doesn't thumbs-down it? That's not reinforcement — that's absence of complaint. When the user thumbs-up? You'll get maybe 5% feedback rates.
- "Observed correctly in 2+ distinct repos" — how do you know it was observed *correctly*? You only know it was *shown*. The user might have ignored it.
- The confidence arithmetic (`+0.12` for cross-project, `+0.1` for promotion) is completely made up. These numbers will need tuning, and you have no data to tune them against.

**My recommendation:** For v1, promotion is manual. The developer runs `/promote lesson_id` or you build a simple dashboard showing lessons that have fired in 2+ repos with positive feedback. Let a human decide. Collect data on what lessons actually travel well. Build the automatic promotion engine in v2 when you have 3+ months of usage data to calibrate the thresholds.

### 3.3 The `causal_chain` Field Is Aspirational

Asking an extraction LLM to produce a reliable "failure → diagnosis → fix arc in 2-3 sentences" is harder than it sounds. In my experience, extraction LLMs produce:
- Generic causal chains: "The build failed because the configuration was wrong. The fix was to change the configuration."
- Hallucinated causal chains that sound plausible but describe a different problem than what actually happened.

**My recommendation:** Keep the field in the schema — it's a good aspiration. But don't *gate* anything on its quality. The `content` field ("When X, do Y") is the actionable part. The causal chain is nice-to-have context. If the extraction produces garbage for `causal_chain`, it shouldn't block the lesson from being stored.

### 3.4 The `extraction_confidence` Score Is Not What You Think

The doc puts a `>= 0.60` quality gate on `extraction_confidence`. But this is a number produced by the extraction LLM itself — you're asking the LLM "how confident are you that you extracted this correctly?" LLMs are notoriously bad at self-calibration. A 0.85 confidence from GPT-4o-mini doesn't mean what you think it means.

**My recommendation:** Don't use the LLM's self-reported confidence as a hard gate. Instead, use structural signals:
- Does the lesson reference at least one concrete file path? (higher quality)
- Is the `source_excerpt` actually present in the transcript? (verifiable)
- Is the lesson type `gotcha` or `convention` rather than `preference`? (more actionable)

Use these as your quality gate. Keep `extraction_confidence` as metadata for analysis, not as a filter.

---

## 4. Edge Cases You're Missing

### 4.1 The Knowledge Graph Staleness Problem

Potpie rebuilds the knowledge graph when a repo is parsed. But the Experience Engine's file anchors are stored as string paths (`docker/Dockerfile`). When files get renamed or moved, every lesson anchored to the old path goes silent. No error, no warning — just a lesson that never fires again.

**What to research:** You need a file-rename detection pass. When the knowledge graph is rebuilt after a parse, diff the file list against anchored lesson paths. Flag lessons whose anchors no longer exist. This is a good Celery background task — `post_parse_health_check`.

### 4.2 Conflicting Lessons

Two developers can produce contradictory lessons: Dev A says "always use moment.js for timezone handling" and Dev B says "never use moment.js." Both get stored, both fire on the same files. The doc mentions `conflict_detection` in passing but provides no implementation.

**My recommendation for v1:** Don't try to auto-detect conflicts. Instead, when showing lessons to the agent, include the developer_id attribution. "Dev A says X. Dev B says Y." Let the agent (and the human) resolve it. Conflict detection is a v2 problem — and it's a hard NLP problem, not a simple cosine-similarity check.

### 4.3 The Feedback Loop Is Weaker Than You Think

The doc assumes developers will regularly thumbs-up/down lessons. In practice, feedback rates on inline suggestions in coding tools are 2-5%. Most sessions will produce zero feedback. Your confidence scores will barely move.

**What to research:** Implicit signals are more reliable than explicit feedback. If the agent was shown a lesson and the task completed without corrections, that's weak positive signal. If the agent was shown a lesson and the developer immediately contradicted it, that's strong negative signal. Parse the transcript for these patterns rather than relying on button clicks.

### 4.4 Token Budget Competition

Potpie agents already have a token budget for conversation history (`get_history_token_budget()` with `HISTORY_MESSAGE_CAP`). Injecting lesson markdown into the system prompt competes with this budget. With 3 file-anchored + 4 semantic lessons, you could easily be injecting 800-1200 tokens of memory context. That's significant.

**My recommendation:** The injection volume cap (3+4 = 7 lessons max) is good, but also enforce a hard token cap — e.g., 600 tokens total for memory injection. Summarize aggressively. The lesson markdown in the doc is verbose ("*Why:* Free runners are capped at 2GB RAM..."). In practice, "GitHub Actions OOM on Docker buildx → use `--memory=4g`" is enough.

---

## 5. The Two Injection Paths — Corrected Architecture

The original review missed a critical design intent: memory should NOT be injected upfront in the system prompt based on the user's vague query. Instead, injection happens at two specific moments during agent execution — when the agent forms an intention, and when the agent touches files.

### 5.1 Why This Matters

User prompts are vague. "Fix the CI" gives you nothing to retrieve against. But when the agent reasons about it and thinks "I need to modify the GitHub Actions workflow to use Docker buildx with layer caching," that chain-of-thought is a precise retrieval signal. Similarly, when the agent calls `get_code_from_probable_node_name("Dockerfile")` or reads a specific file, that's a deterministic anchor for Path A lookup.

The two injection paths correspond to the two retrieval paths, but triggered at the right moment:

```
INJECTION PATH 1 — Chain-of-Thought Interception
──────────────────────────────────────────────────
When:     Agent finishes reasoning, before it commits to a plan
Signal:   Agent's CoT text (technologies mentioned, approach described)
Retrieval: Path B — semantic search + tech-stack overlap
Returns:  "gotcha" and "architecture" lessons relevant to the PLAN
Scope:    developer + project (cross-repo lessons travel here)

INJECTION PATH 2 — File-Touch Interception
──────────────────────────────────────────────────
When:     Agent calls any file-reading tool (get_code, read_file, etc.)
Signal:   Exact file path from the tool call
Retrieval: Path A — deterministic SQLite lookup by file path
Returns:  "gotcha" and "convention" lessons anchored to THAT file
Scope:    project only (file paths are repo-specific)
```

User preferences ("use tabs," "prefer functional style") are different — they're stable, low-churn, and don't need retrieval. These go in the system prompt at session start. They're not lessons; they're configuration.

### 5.2 How CoT Interception Works in Potpie — via `agent.iter()`

**The problem with the `think` tool approach:** Potpie's primary agent — the code gen agent — doesn't use the `think` tool. It goes straight from user prompt to tool calls (knowledge graph queries, code reads, file writes, PR creation). There's no reflective pause to intercept.

**The solution: PydanticAI's `agent.iter()` gives us exactly the interception point we need.**

`agent.iter()` returns an `AgentRun` — an async iterable over the agent's execution graph nodes. The nodes flow in a cycle:

```
UserPromptNode → ModelRequestNode → CallToolsNode → ModelRequestNode → CallToolsNode → ... → End
```

The critical moment is **after `ModelRequestNode` completes but before `CallToolsNode` executes the tools.** At this point, the `CallToolsNode` contains the model's full response — its reasoning text AND its tool call decisions. We can read the model's intent, run Path B retrieval, and inject lessons into the message history before the tools fire.

```python
# integration/iter_interceptor.py
#
# This replaces the think-tool approach. Works with ANY pydantic-ai agent,
# including code_gen_agent which never calls think.

from pydantic_ai._agent_graph import CallToolsNode, ModelRequestNode
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart

async def run_agent_with_memory(
    agent,
    user_prompt: str,
    deps,
    store,           # ExperienceStore / Graphiti wrapper
    project_id: str,
    developer_id: str,
    message_history=None,
):
    memory_injected = False

    async with agent.iter(
        user_prompt,
        deps=deps,
        message_history=message_history,
    ) as agent_run:

        async for node in agent_run:

            # INTERCEPTION POINT: After model responds, before tools execute
            if isinstance(node, CallToolsNode) and not memory_injected:
                model_response: ModelResponse = node.model_response

                # 1. Extract the model's reasoning (TextPart) and tool intentions (ToolCallPart)
                reasoning_text = ""
                intended_files = []
                intended_tech = []

                for part in model_response.parts:
                    if isinstance(part, TextPart):
                        reasoning_text += part.content
                    elif isinstance(part, ToolCallPart):
                        # Tool call args reveal intent:
                        # get_code_from_probable_node_name("Dockerfile")
                        # → the agent intends to touch Docker files
                        intended_files.extend(
                            extract_file_signals(part.tool_name, part.args)
                        )
                        intended_tech.extend(
                            extract_tech_signals(part.tool_name, part.args)
                        )

                # 2. Query Path B using the model's ACTUAL reasoning + intent
                lessons = await store.search_by_plan(
                    plan_text=reasoning_text,
                    tech_stack=intended_tech,
                    file_hints=intended_files,
                    project_id=project_id,
                    developer_id=developer_id,
                )

                # 3. If we found relevant lessons, inject them into message history
                #    BEFORE the tools execute. The model will see these in
                #    its next reasoning step.
                if lessons:
                    memory_block = format_lessons_for_injection(lessons)
                    # Inject as a system-like message that the model sees
                    # when it gets the tool results back
                    agent_run.ctx.state.message_history.append(
                        create_memory_injection_message(memory_block)
                    )

                memory_injected = True  # Only inject once per run

    return agent_run.result
```

**What this gives us:**

```
User: "Fix the CI pipeline"
    │
    ▼
UserPromptNode
    │  (user prompt assembled with system prompt + preferences)
    ▼
ModelRequestNode
    │  LLM reasons: "I need to check the GitHub Actions workflow.
    │  I'll call get_code_from_probable_node_name('build.yml')
    │  and also look at the Dockerfile."
    │
    │  Model response contains:
    │    TextPart: "Let me look at the CI configuration..."
    │    ToolCallPart: get_code_from_probable_node_name("build.yml")
    │    ToolCallPart: get_code_from_node_id("dockerfile_node_123")
    ▼
CallToolsNode  ← WE INTERCEPT HERE
    │
    ├── Extract from model response:
    │   reasoning: "CI configuration...GitHub Actions..."
    │   files: ["build.yml", "Dockerfile"]
    │   tech: ["GitHub Actions", "Docker"]
    │
    ├── Query Path B (Graphiti hybrid search):
    │   → Finds: "GitHub Actions OOM on Docker buildx → use --memory=4g"
    │
    ├── Inject lesson into message history
    │   → Model will see this when processing tool results
    │
    ├── Tools execute normally (get_code returns file contents)
    │   (file_tool_middleware ALSO runs Path A here — double coverage)
    │
    ▼
ModelRequestNode (second loop iteration)
    │  Model now sees: tool results + injected memory
    │  Makes informed decisions with full context
    ▼
... continues until End
```

**Why `agent.iter()` is the right answer:**

- **Works with every pydantic-ai agent.** Code gen, debugging, Q&A, custom agents — all of them flow through the same `UserPromptNode → ModelRequestNode → CallToolsNode` cycle. No dependency on `think` or any specific tool.
- **Zero extra LLM calls.** We intercept between nodes. The LLM has already produced its response. We're reading its output, not asking it to produce more.
- **We see the model's actual intent.** The `CallToolsNode` contains both the reasoning text AND the specific tool calls the model chose. `get_code_from_probable_node_name("Dockerfile")` tells us more about intent than any user prompt ever could.
- **Injection happens at the right moment.** The model sees the lessons when it processes tool results in its next iteration. It hasn't committed to a final approach yet — it's still in its tool-calling loop.
- **Potpie can modify its agents or create new ones.** Since `agent.iter()` is the standard PydanticAI execution primitive, any agent (existing or new) that runs through this wrapper gets memory for free. The execution flows (`StandardExecutionFlow`, `StreamingExecutionFlow`) just need to use `run_agent_with_memory()` instead of `agent.run()`.

**The risk: Streaming compatibility.**

Potpie's primary path is `run_stream()`, not `run()`. The `agent.iter()` approach works naturally with non-streaming execution. For streaming, we need to verify that iterating nodes and streaming tokens can coexist — PydanticAI's `AgentRun` supports this but the streaming integration needs testing with potpie's `RedisStreamManager` and SSE delivery.

**Mitigation:** Phase 1 — implement with `agent.iter()` for the non-streaming path (background/API execution). Phase 2 — adapt for the streaming path once the core logic is proven. The file-tool middleware (Section 5.3) works on both paths immediately since it wraps tool functions, not the execution flow.

### 5.3 How File-Touch Interception Works in Potpie

This is the cleaner path. Potpie's agents use knowledge-graph tools to read code:

- `get_code_from_probable_node_name` — fuzzy file/function lookup
- `get_code_from_node_id` / `get_code_from_multiple_node_ids` — exact node lookup
- `get_code_graph_from_node_id` — dependency graph from a node

Each of these resolves to one or more file paths. We wrap these tools with a middleware layer that:

1. Executes the original tool call (returns the code as normal)
2. Extracts the file path(s) from the tool result
3. Queries the experience engine Path A for lessons anchored to those paths
4. Appends any lessons to the tool response

```
Agent calls get_code_from_probable_node_name("Dockerfile")
    │
    ├── Original tool executes → returns Dockerfile contents
    │
    ├── Middleware extracts file path: "docker/Dockerfile"
    │
    ├── Query experience engine Path A
    │   → POST /retrieve/by-files { file_paths: ["docker/Dockerfile"], project_id }
    │   → Returns: lesson about OOM on buildx (anchored to this file)
    │
    └── Return to agent: original code + appended lesson
        "# docker/Dockerfile
         FROM node:18-alpine ...
         
         ⚠ Lessons anchored to this file:
         - [ESTABLISHED] buildx OOM on free runners → use --memory=4g"
```

**Why this works:**
- Path A is deterministic — SQLite lookup by file path, ~5ms, zero embedding cost.
- Every agent reads files. This is the universal interception point.
- The agent gets the lesson at exactly the moment it's looking at the relevant code.
- No prompt engineering required — it works even if the agent skips `think`.

**The risk: context bloat.** If the agent reads 15 files and each returns 2 lessons, that's 30 injected lesson blocks competing for token budget.

**Mitigation:** Hard cap of 5 file-anchored lessons per session. After 5, the middleware stops appending. Priority: highest confidence first. Track which lessons have already been shown in a session-scoped set so we never repeat.

### 5.4 The Separation: Preferences vs Lessons

| What | Where it lives | When injected | How |
|------|---------------|---------------|-----|
| **Preferences** ("use tabs," "prefer async/await") | System prompt | Session start | Load from developer profile, static per session |
| **Conventions** ("we use date-fns, not moment.js") | System prompt | Session start | Load project conventions, static per session |
| **Gotchas** ("Docker buildx OOMs on free runners") | Tool responses | During execution | Path A via file-touch, Path B via `agent.iter()` interception |
| **Architecture** ("service-auth talks to service-payments via gRPC, not REST") | Tool responses | During execution | Path B via `agent.iter()` interception when cross-service work detected |

Preferences and conventions are cheap — they're a fixed block of 200-400 tokens in the system prompt. They don't need retrieval because they apply to everything the agent does in this repo/developer context.

Gotchas and architecture lessons are expensive and contextual — they should only appear when relevant. That's why they go through the dynamic injection paths, not the system prompt.

---

## 6. Module Structure — Monorepo, Extension-Ready

The experience engine lives as a module inside the potpie monorepo, but with strict boundary rules so it can be extracted to a separate repo/service later.

### 6.1 Folder Layout

```
app/modules/intelligence/memory/experience/
│
├── __init__.py                  # Public API — the ONLY import boundary
│
├── core/                        # The engine itself — zero potpie imports
│   ├── __init__.py
│   ├── schema.py                # Lesson, ExtractionResult, AgentPlan models
│   ├── store.py                 # SQLite + numpy store (Path A + Path B)
│   ├── extractor.py             # Post-mortem LLM extraction
│   └── config.py                # Engine configuration (embedding model, thresholds)
│
├── integration/                 # Potpie-specific wiring — imports from potpie allowed here
│   ├── __init__.py
│   ├── iter_interceptor.py      # agent.iter() interception → Path B retrieval
│   ├── file_tool_middleware.py  # Wraps file-reading tools with Path A lookup
│   ├── preference_loader.py     # Loads developer/project preferences for system prompt
│   ├── extraction_triggers.py   # Celery tasks for post-session extraction
│   └── feedback_handler.py      # Processes thumbs up/down from chat UI
│
└── tests/
    ├── test_store.py
    ├── test_extractor.py
    ├── test_iter_interceptor.py
    └── test_file_middleware.py
```

### 6.2 The Boundary Rule

**`core/` has ZERO imports from `app.modules.*`.** It depends only on stdlib, pydantic, numpy, sqlite3, and an embedding client interface. This is the part that becomes a standalone package when you extract it.

**`integration/` imports from both `core/` and `app.modules.*`.** This is the glue layer. It knows about potpie's `ChatContext`, `ToolService`, `ConversationService`, Celery config. If you extract the engine to a separate repo, this folder stays in potpie as a thin client.

```python
# app/modules/intelligence/memory/experience/__init__.py
#
# This is the public API. Integration code imports from here.
# Nothing outside this module should reach into core/ or integration/ directly.

from .core.store import ExperienceStore
from .core.schema import Lesson, LessonScope, ExtractionResult
from .core.extractor import extract_from_transcript

from .integration.iter_interceptor import run_agent_with_memory
from .integration.file_tool_middleware import wrap_file_tools_with_memory
from .integration.preference_loader import load_session_preferences
from .integration.extraction_triggers import trigger_post_session_extraction
```

### 6.3 How the Pieces Wire Into Potpie

**Wire Point 1: Tool Registration (AgentFactory)**

In `app/modules/intelligence/agents/chat_agents/multi_agent/agent_factory.py`, where tools are assembled for the supervisor agent:

```python
# In AgentFactory.create_supervisor_agent() or wherever tools are composed

from app.modules.intelligence.memory.experience import (
    run_agent_with_memory,
    wrap_file_tools_with_memory,
)

# 1. Use run_agent_with_memory() instead of agent.run() directly
#    This wraps the execution with agent.iter() interception for Path B

# 2. Wrap file-reading tools with Path A middleware
wrapped_tools = wrap_file_tools_with_memory(
    original_tools=tools,  # the existing tool list
    store=experience_store,
    project_id=ctx.project_id,
    tool_names_to_wrap=[
        "get_code_from_probable_node_name",
        "get_code_from_node_id",
        "get_code_from_multiple_node_ids",
        "get_code_graph_from_node_id",
    ],
)
```

**Wire Point 2: System Prompt (Preferences)**

In the execution flows, before creating the supervisor agent:

```python
from app.modules.intelligence.memory.experience import load_session_preferences

# In StandardExecutionFlow.run() or the prompt assembly step
preferences = load_session_preferences(
    project_id=ctx.project_id,
    developer_id=ctx.user_id,
)
if preferences:
    # Append to the system prompt alongside existing context
    system_prompt += f"\n\n## Developer Preferences\n{preferences}"
```

**Wire Point 3: Extraction Trigger (Celery)**

In `app/celery/tasks/` alongside existing agent tasks:

```python
# app/celery/tasks/memory_tasks.py

from app.modules.intelligence.memory.experience import trigger_post_session_extraction

@shared_task(name="experience.extract_post_session")
def extract_post_session(conversation_id: str, project_id: str, user_id: str):
    """
    Fired when a conversation ends (idle timeout or explicit stop)
    or when a PR is created via the GitHub tools.
    """
    trigger_post_session_extraction(
        conversation_id=conversation_id,
        project_id=project_id,
        developer_id=user_id,
    )
```

This task gets triggered from two places:
- `ConversationService` when a conversation is stopped or goes idle
- The `github_create_pull_request` tool's success callback

**Wire Point 4: Feedback (API Route)**

A new lightweight route in the conversations router or a new router:

```python
# In conversations_router.py or a new experience_router.py

@router.post("/experience/feedback/{lesson_id}")
async def lesson_feedback(lesson_id: str, helpful: bool, user_id: str = Depends(get_current_user)):
    experience_store.record_feedback(lesson_id, helpful)
    return {"status": "recorded"}
```

### 6.4 The Complete Execution Flow

```
User: "Fix the CI pipeline"
    │
    ▼
ConversationService receives message
    │
    ├── Resolves node_ids from knowledge graph (EXISTING)
    ├── Loads developer preferences → injects into system prompt
    │
    ▼
PydanticMultiAgent.run_stream(ctx)
    │
    ▼
Supervisor agent starts reasoning...
    │
    ├── ModelRequestNode: LLM reasons about the CI failure,
    │   decides to call get_code_from_probable_node_name("build.yml")
    │   │
    ├── agent.iter() intercepts at CallToolsNode (before tools execute):
    │       ├── Reads model's reasoning + tool call intentions
    │       ├── Queries Path B: semantic search across developer-scoped lessons
    │       └── Injects: "⚠ OOM lesson from service-auth" into message history
    │                            ← INJECTION PATH 1
    │
    ├── Agent (now aware of OOM risk) calls
    │   get_code_from_probable_node_name("build.yml")
    │   │
    │   └── file_tool_middleware intercepts:
    │       ├── Original tool returns: file contents
    │       ├── Extracts file path: ".github/workflows/build.yml"
    │       ├── Queries Path A: exact match on file_anchor_index
    │       └── Returns: code + "⚠ --memory=4g required for buildx"
    │                            ← INJECTION PATH 2
    │
    ├── Agent modifies the file WITH the lessons applied
    │   (includes --memory=4g, uses build cache)
    │
    ▼
Agent creates PR via github_create_pull_request
    │
    ├── Success callback: extract_post_session.delay(...)
    │                            ← EXTRACTION TRIGGER
    │
    ▼
[Background Celery task]
    ├── Fetches conversation transcript from MessageStore
    ├── Sends to extraction LLM
    ├── Stores any new lessons with file_anchors + tech_stack
    └── New lessons available for future sessions
```

---

## 7. Critique of the Injection Approach — Honest Risks

### 7.1 Streaming Compatibility for `agent.iter()`

The `agent.iter()` interception works naturally with non-streaming execution. Potpie's primary path is `run_stream()` via `StreamingExecutionFlow`. We need to verify that node-level iteration and token streaming can coexist — PydanticAI's `AgentRun` supports both, but the integration with potpie's `RedisStreamManager` and SSE delivery needs testing.

**Mitigation for v1:** Implement `agent.iter()` interception for the non-streaming path first (background/API execution via `StandardExecutionFlow`). The file-tool middleware (Path 2) works on both streaming and non-streaming paths immediately. Adapt the streaming path in Phase 2 once core logic is proven.

### 7.2 Tool Response Bloat

Appending lesson text to tool responses inflates the context window at the tool-result level, not the system-prompt level. This means every subsequent model call in the agent loop carries those extra tokens.

**Mitigation:** Keep lesson injection terse — 1-2 lines per lesson, not the verbose markdown from the design doc. A session-scoped dedup set ensures the same lesson never appears twice. Hard cap: 5 lessons total injected via tool responses per session.

### 7.3 Extraction Quality Is Still the Biggest Unknown

No matter how elegant the injection architecture is, it's only as good as the lessons in the store. If the extraction LLM hallucinates, the agent gets bad advice at exactly the wrong moment — when it's reading a critical file.

**Pre-launch requirement:** Before wiring this into production, manually run 20 real conversation transcripts through the extraction prompt. Grade every extracted lesson. If < 60% are correct and actionable, iterate on the extraction prompt before building the rest.

### 7.4 The Middleware Must Be Invisible on Failure

If the experience engine's SQLite is corrupted, if the embedding service is down, if the store query times out — the agent must still work normally. The middleware must fail silently with a log line, never with an exception that breaks the tool call.

```python
# In file_tool_middleware.py — the pattern for every middleware call
try:
    lessons = store.get_by_file_paths(file_paths, project_id)
    if lessons:
        result += format_lessons_for_tool_response(lessons)
except Exception as e:
    logger.warning(f"Experience engine unavailable: {e}")
    # Return the original tool result unchanged
```

---

## 8. v1 Scope — What to Build, In Order

### Phase 1: The Store + File Interception (Week 1-2)

Build `core/schema.py`, `core/store.py`, and `integration/file_tool_middleware.py`. This gives you Path A — deterministic file-anchored retrieval on every file read. Manually seed 10-20 lessons from real conversations to test retrieval quality. No extraction LLM yet — humans write the lessons.

**Why this first:** It's the lowest-risk, highest-signal injection path. Zero embedding cost, zero LLM calls in the hot path, and it catches the most dangerous class of bugs (file-specific gotchas).

### Phase 2: The Think Tool Wrapper (Week 2-3)

Build `integration/iter_interceptor.py` using `agent.iter()` to intercept after `CallToolsNode`. This gives you CoT interception — reading the model's reasoning + tool intentions to trigger Path B retrieval.

**Why second:** This depends on an embedding service and has higher latency (~100-200ms). Test it with real agent sessions before making it the default.

### Phase 3: Extraction + Triggers (Week 3-4)

Build `core/extractor.py` and `integration/extraction_triggers.py`. Wire into Celery. Start with one trigger: conversation end (idle timeout or explicit stop). Add PR creation trigger in the same week if the first one works.

**Why third:** Extraction quality is the biggest risk. You need the store and injection paths working first so you can actually test whether extracted lessons fire correctly.

### Phase 4: Preferences + Feedback (Week 4-5)

Build `integration/preference_loader.py` and `integration/feedback_handler.py`. Add the feedback API route. Wire preferences into the system prompt.

**Why last:** Preferences are the simplest feature — static injection at session start. Feedback is the least urgent — the system works without it, and you'll learn more from watching agent behavior than from thumbs up/down rates.

### Defer to v2:

- ORG scope and multi-tenant promotion
- Automatic promotion engine (manual promotion only in v1)
- Conflict detection between lessons
- The `/reflect` explicit command
- Tests-green trigger
- Extracting `core/` into a standalone package

---

## 9. Research Items — Validate Before Committing

### 9.1 Extraction Quality

Run 20 real potpie conversation transcripts through the extraction prompt. Manually grade: what % of extracted lessons are actually correct and actionable? If it's below 60%, the extraction prompt needs iteration before you build the store.

### 9.2 `agent.iter()` Integration with Streaming

Test whether the `agent.iter()` node interception can coexist with potpie's `run_stream()` path. Specifically: can you iterate nodes via `AgentRun` while simultaneously streaming tokens through `RedisStreamManager` → SSE? If not, the `agent.iter()` approach is limited to non-streaming (background/API) execution, and streaming relies solely on file-tool middleware (Path 2).

### 9.3 Retrieval Precision at Developer Scope

The tech-stack overlap heuristic (`["Docker", "GitHub Actions"]` matching across repos) has a precision risk. "Docker" is used in 80% of repos. A Docker lesson from repo A will fire on every Docker-using repo. Test this with 5+ repos from a single developer and measure false-positive rate. You may need 3+ tech anchors for developer scope.

### 9.4 Latency Budget

The embedding call to OpenAI's `text-embedding-3-small` has p95 latency of ~200-400ms depending on region. For the `agent.iter()` interception, this adds to the time between model response and tool execution. Measure it. Consider a local embedding model (`all-MiniLM-L6-v2` via sentence-transformers, ~10ms on CPU) for v1 if OpenAI latency blows your budget.

### 9.5 Tool Response Token Overhead

Measure the actual token cost of appended lessons in tool responses across 50 sessions. How much does it inflate total token usage? Is it < 5% overhead (acceptable) or > 15% (needs aggressive truncation)?

---

## 10. Standing on Shoulders — Leveraging Graphiti & Letta

The junior's PoC originally proposed Graphiti for storage. The v2 critique rejected it for cost ($180 to bootstrap, LLM calls on every write). After deeper analysis, the right answer is in between: **use Graphiti as the storage + retrieval engine, adopt Letta's memory model as the design pattern, but don't adopt either's runtime.**

Both projects are genuinely excellent at specific things. The goal is to compose their strengths while avoiding the parts that don't fit a coding-agent memory system.

### 10.1 What Graphiti Gives Us (Use as Library, Not Fork)

**Graphiti is `pip install graphiti-core`.** It runs on your existing Neo4j instance. No new infrastructure.

**What we get for free that would otherwise take weeks to build:**

**Hybrid search.** Graphiti combines semantic similarity (cosine on embeddings), BM25 keyword search, and graph traversal — then reranks with a cross-encoder. This is categorically better than raw numpy cosine similarity for Path B retrieval. When the agent thinks "I need to modify the GitHub Actions workflow for Docker buildx," Graphiti's hybrid search finds the OOM lesson even if the lesson text says "CI runner memory limit" instead of "Docker buildx" — because BM25 catches "GitHub Actions" as a keyword match while semantic search catches the intent overlap. Building this hybrid search + reranking ourselves would be 3-4 weeks of work. Graphiti ships it.

**Temporal invalidation.** Graphiti tracks fact validity windows (`t_valid`, `t_invalid`) natively. Our per-project invalidation maps directly: when a lesson is invalidated in `service-payments`, we set `t_invalid` on that project's edge. The lesson remains valid (edge still has `t_valid`, no `t_invalid`) in every other project. This is exactly the bitemporal model Graphiti was designed for.

**Entity deduplication.** When we store a lesson about "Docker" and another about "Docker BuildKit," Graphiti's entity resolution recognizes these as related and links them in the graph. This means a search for "Docker" also surfaces lessons connected to "Docker BuildKit" via graph traversal — something pure vector search would miss.

**Custom entity types via Pydantic.** Graphiti lets you define domain-specific entity types as Pydantic models. We define `Lesson`, `FileAnchor`, `TechStackTag` as custom types. Graphiti handles schema validation, attribute population, and typed storage. Our lesson schema from Section 5 maps directly.

**The group_id mechanism.** Graphiti's `group_id` parameter on episodes creates natural namespaces. `group_id=f"project_{project_id}"` for project-scoped lessons, `group_id=f"developer_{developer_id}"` for developer-scoped lessons. Search is scoped to `group_ids`, which maps perfectly to our scope hierarchy.

**What we DON'T use from Graphiti:**

**The automatic entity extraction during ingestion.** Graphiti calls GPT-4o-mini during `add_episode()` to extract entities and relationships from raw text. This is designed for unstructured data ("parse this customer conversation and figure out the entities"). We don't need this — our post-mortem extractor already produces structured `Lesson` objects with explicit `file_anchors` and `tech_stack_anchors`. We feed pre-structured data into Graphiti.

This is a known pattern. Multiple Graphiti users have asked for "bring your own extraction" (GitHub issues #1193, #1299). The approach: define custom entity types so Graphiti knows the schema, then feed it structured episodes where the entities are already explicit. This minimizes internal LLM calls to only deduplication and edge resolution — not full extraction.

**Risk to validate:** Test whether `add_episode` with fully-specified custom entity types actually reduces LLM calls to near-zero, or whether Graphiti still runs its extraction pipeline regardless. If the latter, write a thin adapter that uses Graphiti's search APIs but bypasses ingestion for direct Neo4j writes of pre-structured lesson nodes. This is a day of testing, not a week of architecture.

### 10.2 What Letta Gives Us (Use as Design Pattern, Not Runtime)

Letta (formerly MemGPT) is a full agent runtime — the agent loop, tool execution, memory tiers, state persistence. Potpie already has all of this. Adopting Letta's runtime would mean replacing `PydanticMultiAgent`, `ConversationService`, and the entire execution flow. That's not happening.

But Letta's **memory model** is the best-designed pattern in the space. We adopt the pattern, not the platform.

**The three-tier memory model, mapped to our system:**

| Letta Tier | Letta Description | Our Implementation |
|-----------|-------------------|-------------------|
| **Core Memory** | Small, persistent blocks always in the context window. Agent can edit them. Contains persona, user details, critical context. | **Developer preferences + project conventions.** Loaded at session start, injected into system prompt. Static per session. The agent doesn't edit these — they come from the experience store. |
| **Recall Memory** | Searchable conversation history stored outside context. Agent queries it when looking back at prior interactions. | **Existing conversation history in potpie.** Already implemented via `ChatHistoryService` and `MessageStore`. No change needed. |
| **Archival Memory** | Long-term searchable storage for facts and knowledge. Agent queries via tool calls. Scales to millions of entries. | **The Experience Engine lesson store in Graphiti.** Lessons are searched via `agent.iter()` interception (Path B) and file-tool middleware (Path A). |

**The self-editing pattern:**

Letta's key innovation is that agents can *write* to memory, not just read from it. In Letta, agents call `archival_memory_insert` to store facts and `core_memory_replace` to update their persona.

We adopt this as an explicit tool the agent can call during a session:

```
Tool: remember_lesson
Description: "Store an explicit rule or lesson for future sessions.
             Use this when the user states a project rule, when you
             discover a non-obvious constraint, or when a workaround
             is required."
Input: { content: str, file_anchors: list[str], tech_stack: list[str] }
```

This complements the post-mortem extraction. The post-mortem catches lessons the agent didn't explicitly recognize. The `remember_lesson` tool catches lessons the agent *does* recognize in the moment — "the user just told me to never use moment.js, I should store this."

**What we DON'T use from Letta:**

- **The runtime.** Our agents run in PydanticAI via `PydanticMultiAgent`. Letta's agent loop is not involved.
- **The `ai-memory-sdk`.** It creates a Letta agent under the hood for memory management. Overkill — we just need the pattern.
- **Core memory self-editing.** In Letta, agents update their own persona and user blocks. Our preferences are developer-managed, not agent-managed. The agent reads them but doesn't change them. Agent-managed preferences are a v3 feature after trust is established.

### 10.3 Updated Module Structure — With Graphiti & Letta Patterns

```
app/modules/intelligence/memory/experience/
│
├── __init__.py                    # Public API
│
├── core/                          # Engine — depends on graphiti-core, not on potpie
│   ├── __init__.py
│   ├── schema.py                  # Pydantic models: Lesson, FileAnchor, TechStackTag
│   │                              # (registered as Graphiti custom entity types)
│   │
│   ├── graphiti_store.py          # Wraps graphiti-core client
│   │   ├── init_graphiti()        # Connect to existing Neo4j instance
│   │   ├── store_lesson()         # add_episode with pre-structured Lesson entity
│   │   ├── search_by_plan()       # Hybrid search (Path B) — semantic + BM25 + graph
│   │   ├── search_by_files()      # Graph traversal from FileAnchor → Lesson (Path A)
│   │   ├── invalidate_lesson()    # Set t_invalid on project-scoped edge
│   │   └── record_feedback()      # Update confidence via metadata
│   │
│   ├── memory_blocks.py           # Letta-inspired core memory blocks
│   │   ├── load_preferences()     # Developer preferences → system prompt block
│   │   ├── load_conventions()     # Project conventions → system prompt block
│   │   └── format_blocks()        # Render blocks as markdown for injection
│   │
│   ├── extractor.py               # Post-mortem extraction (our prompt, our LLM)
│   │                              # Outputs structured Lesson → feeds into graphiti_store
│   └── config.py                  # Graphiti connection, embedding model, thresholds
│
├── integration/                   # Potpie-specific wiring
│   ├── __init__.py
│   ├── iter_interceptor.py      # CoT interception → graphiti_store.search_by_plan()
│   ├── file_tool_middleware.py    # File read → graphiti_store.search_by_files()
│   ├── remember_tool.py           # Letta-inspired: agent explicitly stores a lesson
│   │                              # during session → graphiti_store.store_lesson()
│   ├── extraction_triggers.py     # Celery tasks for post-session extraction
│   └── feedback_handler.py        # Thumbs up/down → graphiti_store.record_feedback()
│
└── tests/
    ├── test_graphiti_store.py     # Test against a test Neo4j instance
    ├── test_extractor.py
    ├── test_iter_interceptor.py
    ├── test_file_middleware.py
    └── test_remember_tool.py
```

### 10.4 How the Pieces Compose — Graphiti + Letta Pattern + Potpie

```
Session Start
    │
    ├── memory_blocks.load_preferences(developer_id)     ← LETTA PATTERN
    │   Returns: "Prefers async/await, uses 2-space indent, hates moment.js"
    │
    ├── memory_blocks.load_conventions(project_id)        ← LETTA PATTERN
    │   Returns: "This repo uses FastAPI + SQLAlchemy, tests in pytest"
    │
    ├── Both injected into system prompt as core memory blocks
    │
    ▼
Agent Reasoning Phase
    │
    ├── ModelRequestNode completes → model reasons about Docker workflow
    │   CallToolsNode contains: TextPart("I'll modify the workflow...")
    │                          + ToolCallPart(get_code_from_probable_node_name, "build.yml")
    │   │
    │   └── iter_interceptor intercepts CallToolsNode:
    │       ├── graphiti_store.search_by_plan(                ← GRAPHITI SEARCH
    │       │     plan_text="modify Docker workflow...",
    │       │     group_ids=["project_X", "developer_Y"],
    │       │     tech_stack=["Docker", "GitHub Actions"]
    │       │   )
    │       │   Graphiti runs: semantic + BM25 + graph traversal + rerank
    │       │   Returns: OOM lesson (found via BM25 keyword "GitHub Actions"
    │       │            + semantic match on "Docker build" + graph link
    │       │            from TechStackTag:Docker → Lesson)
    │       │
    │       └── Injects lesson into message history before tools execute
    │
    ▼
Agent File Reading Phase
    │
    ├── Agent calls get_code_from_probable_node_name("Dockerfile")
    │   │
    │   └── file_tool_middleware:
    │       ├── Original tool returns file contents
    │       ├── graphiti_store.search_by_files(                ← GRAPHITI SEARCH
    │       │     file_paths=["docker/Dockerfile"],
    │       │     group_ids=["project_X"]
    │       │   )
    │       │   Graphiti traverses: FileAnchor:"docker/Dockerfile" → Lesson
    │       │   Deterministic graph lookup, ~5ms
    │       │
    │       └── Returns code + anchored lessons
    │
    ▼
Agent Discovers Explicit Rule
    │
    ├── User says: "We never use alpine images, they break our C extensions"
    │   │
    │   └── Agent calls remember_lesson(                      ← LETTA PATTERN
    │         content="Never use alpine base images — C extensions break",
    │         file_anchors=["docker/Dockerfile"],
    │         tech_stack=["Docker"]
    │       )
    │       │
    │       └── graphiti_store.store_lesson()                  ← GRAPHITI WRITE
    │           Structured Lesson → add_episode with custom entity types
    │           Graphiti handles: embedding, dedup, graph linking
    │           Available immediately for future sessions
    │
    ▼
Session Ends / PR Created
    │
    ├── Celery task: extraction_triggers.extract_post_session()
    │   │
    │   ├── extractor.py runs post-mortem on transcript
    │   │   (our extraction prompt, our LLM — NOT Graphiti's)
    │   │
    │   ├── For each extracted lesson:
    │   │   graphiti_store.store_lesson(lesson)                ← GRAPHITI WRITE
    │   │   Graphiti checks for near-duplicates via entity resolution
    │   │   If duplicate found: compounds confidence on existing lesson
    │   │   If novel: creates new Lesson episode with file/tech anchors
    │   │
    │   └── Lessons available for all future sessions
```

### 10.5 Graphiti-Specific Custom Entity Schema

```python
# core/schema.py — Pydantic models registered as Graphiti custom entity types

from pydantic import BaseModel, Field
from typing import Literal


class Lesson(BaseModel):
    """An actionable engineering lesson learned from a coding session.
    Starts with 'When X, do Y' or 'Never Z because W'."""

    content: str = Field(description="The actionable rule")
    lesson_type: Literal["gotcha", "convention", "preference", "architecture"]
    causal_chain: str = Field(default="", description="Why this lesson exists")
    confidence: float = Field(default=0.4, ge=0.0, le=1.0)
    times_reinforced: int = Field(default=0)
    scope: Literal["project", "developer"] = "project"
    origin_project_id: str = Field(default="")
    source_excerpt: str = Field(default="", description="2-4 lines from the conversation")


class FileAnchor(BaseModel):
    """A specific file path in a repository. Lessons connect to FileAnchors
    when they are relevant to that file."""

    file_path: str = Field(description="e.g. 'docker/Dockerfile'")
    project_id: str = Field(description="Which project this file belongs to")


class TechStackTag(BaseModel):
    """A technology, framework, or tool. Used for cross-repo lesson retrieval."""

    name: str = Field(description="e.g. 'Docker', 'GitHub Actions', 'FastAPI'")


# Graphiti entity type registration
entity_types = {
    "Lesson": Lesson,
    "FileAnchor": FileAnchor,
    "TechStackTag": TechStackTag,
}

# Edge type map — which relationships are valid between entity types
edge_type_map = {
    ("Lesson", "FileAnchor"): ["ANCHORED_TO"],
    ("Lesson", "TechStackTag"): ["INVOLVES_TECH"],
    ("FileAnchor", "TechStackTag"): ["USES_TECH"],
}
```

### 10.6 What This Gives Us Over Raw SQLite + Numpy

| Dimension | SQLite + Numpy (original v2) | Graphiti on Neo4j |
|-----------|------------------------------|-------------------|
| **Path A retrieval** | SQLite index on file_path. ~5ms. | Graph traversal: FileAnchor → Lesson. ~5ms. Equivalent. |
| **Path B retrieval** | Numpy cosine similarity. Single signal. | Hybrid: semantic + BM25 + graph traversal + reranking. Three signals fused. Significantly better recall. |
| **Deduplication** | Manual cosine check (> 0.90 = same lesson). | Graphiti entity resolution — handles paraphrases, synonyms, near-duplicates across entity types. |
| **Cross-repo linking** | Tech-stack string overlap heuristic. | Graph edges: Lesson → TechStackTag → (other Lessons with same tag). Graph traversal finds connections string matching misses. |
| **Temporal invalidation** | `invalidated_in_projects` JSON dict. Works but manual. | Bitemporal model: `t_valid`, `t_invalid` per edge. Built-in temporal queries. |
| **Scale ceiling** | Numpy full scan breaks at ~10K lessons. | Neo4j indexes + Graphiti's Lucene-backed BM25. Near-constant time regardless of graph size. |
| **New infrastructure** | SQLite file. Zero ops. | Uses your **existing** Neo4j instance. Zero new infrastructure. |
| **Dependency cost** | Zero. | `pip install graphiti-core`. Maintained by Zep (23K+ GitHub stars, active development). |

### 10.7 Risks & Mitigations for the Graphiti + Letta Approach

**Risk 1: Graphiti's internal LLM calls on write.**

Even with custom entity types, `add_episode` may still trigger entity extraction and edge dedup LLM calls internally. At scale (50+ lessons per project), this adds up in cost and latency.

**Mitigation:** Week 1 deliverable — write a test that calls `add_episode` with a fully pre-structured Lesson entity and measures: (a) how many LLM calls Graphiti makes internally, (b) total write latency. If the answer is >2 LLM calls or >2s latency, build an adapter that writes Lesson nodes directly to Neo4j via Cypher and uses Graphiti *only* for search. The search path never calls LLMs — it's pure embedding + BM25 + graph traversal.

**Risk 2: Graphiti version churn.**

Graphiti is actively developed (v3 just launched, breaking changes from v2). Depending on it means tracking their releases.

**Mitigation:** The `graphiti_store.py` wrapper in `core/` is our insulation layer. All Graphiti API calls go through this one file. If Graphiti changes their API, we update one file. If we ever need to replace Graphiti entirely, we swap the implementation behind the same interface.

**Risk 3: Neo4j namespace collision.**

Potpie's knowledge graph (files, functions, classes, imports) lives in the same Neo4j instance. Graphiti's nodes (Lessons, FileAnchors, TechStackTags) must not collide.

**Mitigation:** Graphiti uses `group_id` for namespace isolation. All experience engine episodes use `group_id` prefixed with `exp_` (e.g., `exp_project_123`, `exp_developer_456`). Potpie's knowledge graph nodes don't use Graphiti's label scheme (`Entity`, `Episode`). The two graph structures coexist in the same Neo4j instance without interference. Validate this in the first week with a smoke test.

**Risk 4: The `remember_lesson` tool gets misused.**

The agent might call `remember_lesson` for trivial things ("user prefers dark mode") or hallucinate lessons that were never stated. This pollutes the store.

**Mitigation:** The tool's response includes a confirmation: "Lesson stored with `pending` status. It will be reviewed." All agent-authored lessons start at `confidence=0.3` and `approval_status=pending`. They only fire in future sessions after a human approves them (or after the post-mortem extraction independently extracts the same lesson, which auto-approves it via Graphiti's dedup).

---

## 11. Updated v1 Scope — With Graphiti & Letta Patterns

### Phase 1: Graphiti Foundation + File Interception (Week 1-2)

- Set up `graphiti-core` pointing at existing Neo4j
- Define custom entity types: `Lesson`, `FileAnchor`, `TechStackTag`
- Build `graphiti_store.py` with search_by_files (Path A via graph traversal)
- Build `file_tool_middleware.py` wrapping potpie's code-reading tools
- **Validate:** LLM call count on `add_episode` with pre-structured entities
- Manually seed 10-20 lessons to test retrieval quality

### Phase 2: Hybrid Search + Think Tool (Week 2-3)

- Build `graphiti_store.search_by_plan()` using Graphiti's hybrid search
- Build `iter_interceptor.py` using `agent.iter()` for CoT interception
- Build `memory_blocks.py` for Letta-pattern preference injection
- **Validate:** Hybrid search precision vs numpy-only baseline on 50 test queries

### Phase 3: Extraction + Remember Tool (Week 3-4)

- Build `extractor.py` with the post-mortem extraction prompt
- Build `extraction_triggers.py` as Celery tasks (conversation end + PR creation)
- Build `remember_tool.py` — agent can explicitly store lessons mid-session
- Wire extraction output → `graphiti_store.store_lesson()`
- **Validate:** Extraction quality on 20 real transcripts (target: >60% correct)

### Phase 4: Feedback + Confidence (Week 4-5)

- Build `feedback_handler.py` — thumbs up/down updates confidence in Graphiti
- Build confidence compounding: when Graphiti dedup detects a near-duplicate, compound confidence on the existing lesson
- Add feedback API route to potpie
- **Validate:** End-to-end flow — lesson extracted → stored → retrieved → feedback → confidence updated

### Defer to v2:

- ORG scope and multi-tenant promotion
- Automatic promotion engine
- Conflict detection between lessons
- Agent-managed preference editing (Letta's full core memory self-editing)
- The `/reflect` explicit command
- Tests-green extraction trigger

---

## 12. Summary

The Experience Engine combines three sources of strength:

**From the v2 design doc:** Two-path retrieval (file-anchored + semantic), scope hierarchy (project + developer), per-project invalidation, async post-mortem extraction, and the two injection moments (CoT interception via `agent.iter()` node stepping + file-touch middleware).

**From Graphiti:** Hybrid search (semantic + BM25 + graph traversal + reranking), temporal invalidation via bitemporal edges, entity deduplication and resolution, custom Pydantic entity types, and zero new infrastructure (runs on potpie's existing Neo4j).

**From Letta:** The three-tier memory model (core/recall/archival mapped to preferences/history/lessons), the self-editing pattern (agent can explicitly store lessons via the `remember_lesson` tool), and the clean separation between always-in-context memory (preferences) and on-demand retrieval (lessons).

For v1: Graphiti as `pip install` dependency (not fork), Letta as design influence (not runtime dependency), module inside the potpie monorepo with strict `core/` vs `integration/` boundary. Validate Graphiti's write-path LLM cost in week 1. Validate extraction quality before trusting automatic lesson creation. Ship file-anchored retrieval first — it's the highest-signal, lowest-risk path.