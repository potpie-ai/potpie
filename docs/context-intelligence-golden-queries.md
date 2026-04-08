# Context Intelligence Layer — Golden Query Set

Use this set to evaluate `resolve_context` (HTTP `POST .../query/resolve-context`, MCP `context_resolve`, or QnA with `CONTEXT_INTELLIGENCE_ENABLED=true`). Substitute real values for your project:

- `<PROJECT_ID>` — UUID from your environment
- `<PR_N>` — a PR you have ingested (e.g. `694`)
- `<FILE>` — a path that exists in the graph (e.g. `app/main.py`)

---

## How to score a run

For each question, check:

| Check | Meaning |
|-------|---------|
| **Signals** | `mentioned_pr`, paths, `needs_history`, etc. look reasonable (inspect bundle or logs if available). |
| **Plan vs bundle** | Expected evidence families (semantic, artifact, changes, decisions, discussions, ownership) are populated when data exists. |
| **Coverage** | `coverage.status` matches expectation (`complete` / `partial` / `empty`). |
| **Renderer** | If using Potpie, the `CONTEXT INTELLIGENCE (PREFETCHED)` block is present and readable. |
| **Tool churn** | With prefetch on, the model should not chain duplicate `get_project_context` / `get_decisions` / `get_change_history` when coverage is complete. |

---

## 1. PR rationale (why / intent)

These should strongly pull **artifact**, **discussions**, **decisions**, and **change_history** when a PR is mentioned or rationale keywords appear.

1. Why was the retry policy added in PR #`<PR_N>`?
2. What problem did PR #`<PR_N>` solve?
3. Summarize the main rationale for merging PR #`<PR_N>`.
4. What alternatives were discussed before merging PR #`<PR_N>`?

---

## 2. Review and discussion

These stress **discussion_context** and **artifact_context** for a known PR.

1. What was discussed in PR #`<PR_N>`?
2. List review threads or line comments for PR #`<PR_N>`.
3. What feedback did reviewers leave on PR #`<PR_N>`?
4. Were there unresolved concerns in PR #`<PR_N>`?

---

## 3. File history and changes

These stress **change_history** (and optionally **decisions**) with a **file path** in the query.

1. Who changed `<FILE>` recently?
2. What PRs modified `<FILE>`?
3. Show recent change history for `<FILE>`.
4. What is the history of changes around `<FILE>`?

---

## 4. Ownership

These stress **ownership_context** when both ownership cues and a file path appear.

1. Who maintains or owns `<FILE>`?
2. Who worked on `<FILE>` most in recent PRs?
3. Which developers touched `<FILE>`?

---

## 5. Semantic / episodic recall (no specific PR)

These stress **semantic_search** when episodic layer is enabled — no `#PR` required.

1. What decisions were recorded about error handling in this codebase?
2. Summarize what the context graph knows about authentication changes.
3. Find anything in project memory related to webhooks.

---

## 6. Code navigation (should not over-fetch graph history)

These questions are phrased like “what does X do” **without** history keywords or PR numbers. The layer should **not** behave like every turn needs full PR/decision/discussion fetches; signals may mark **code navigation** and keep structural/semantic load lighter than PR-deep queries.

1. What does the main entrypoint of this service do?
2. Where is the webhook handler implemented?
3. Explain how request routing works in this project.
4. What does the function `parse_directory` do? *(adjust name to match your repo)*

**Pass criteria:** Resolution still returns a sensible bundle; coverage may be partial or semantic-only; agent should rely on code tools for implementation detail, not only the graph.

---

## 7. Edge and negative cases

1. What was discussed in PR #999999? *(expect empty or partial; no crash)*
2. Show decisions for file `nonexistent/path/that/does/not/exist.py`.
3. Summarize PR #`<PR_N>` when the PR exists but review threads are empty *(partial coverage is OK).*

---

## 8. HTTP/MCP smoke (same questions, API contract)

Repeat 2–3 questions from sections 1–4 against:

- `POST /query/resolve-context` with body:
  - `project_id`, `query`, optional `timeout_ms`, optional `artifact` / `scope`

Confirm the response includes top-level **`coverage`**, **`errors`**, **`meta`** (with `schema_version`) and **`bundle`**.

---

## Suggested minimum run (10 queries)

If you only run ten:

1. PR rationale — *Why was X done in PR #`<PR_N>`?*
2. Review — *What was discussed in PR #`<PR_N>`?*
3. File history — *What PRs changed `<FILE>`?*
4. Ownership — *Who likely owns `<FILE>`?*
5. Semantic — *What does the graph recall about [topic]?*
6. Code nav — *What does [component] do?* (no PR)
7. Edge — *PR that does not exist*
8. Edge — *empty review threads on a real PR* (if applicable)
9. Mixed — *For PR #`<PR_N>`, summarize rationale and main review themes.*
10. Mixed — *For `<FILE>`, summarize owners and recent changes.*

---

## Related

- Broader tool-level prompts: [context-tool-test-questions.md](./context-tool-test-questions.md)
- Spec: [context-graph-intelligence-layer-spec.md](./implementation-plans/context-graph-intelligence-layer-spec.md)
