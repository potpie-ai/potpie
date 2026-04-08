## Context Intelligence Layer — Cal.com PR #28479 Test Report

- **Project**: Cal.com (`calcom/cal.com`)
- **Project ID**: `019d430a-0518-7769-98e9-f32f3b920fa4`
- **PR under test**: `#28479` — `fix: use i18n for apps count with proper pluralization`
- **Provider**: `HybridGraphIntelligenceProvider`
- **Timeout budget**: `4000ms` per `resolve_context` call

This report evaluates the context-intelligence layer (not the full chat agent) for a focused set of PR-centric queries around Cal.com PR `#28479`.

---

## 1. Queries and coverage

All queries used:

```json
POST /api/v1/context/query/resolve-context
{
  "project_id": "019d430a-0518-7769-98e9-f32f3b920fa4",
  "query": "<QUESTION>",
  "consumer_hint": "<ID>",
  "timeout_ms": 4000
}
```

Exact questions:

```json
[
  {
    "id": "Q1_pr_summary",
    "query": "For Cal.com, summarize GitHub PR #28479: what does it change, and what problem is it trying to solve?"
  },
  {
    "id": "Q2_pr_rationale",
    "query": "Why was PR #28479 needed in Cal.com? Explain the main rationale for merging it."
  },
  {
    "id": "Q3_pr_risk",
    "query": "For PR #28479 in Cal.com, what are the most likely regressions and user-visible risks introduced by this change?"
  },
  {
    "id": "Q4_pr_related",
    "query": "List other PRs and decisions that are closely related to PR #28479 in Cal.com (same files, feature area, or follow-up fixes)."
  },
  {
    "id": "Q5_debug_scenario",
    "query": "Suppose after deploying PR #28479, booking confirmations intermittently fail for recurring events. Using PR #28479 and nearby changes, outline a concrete debug plan."
  }
]
```

### 1.1 Summary table

| ID | Type | Query (paraphrased) | HTTP | Coverage.status | Families available | Errors |
| --- | --- | --- | --- | --- | --- | --- |
| **Q1_pr_summary** | PR summary | Summarize PR #28479: what it changes and what problem it solves | 200 | **complete** | `semantic_search`, `artifact_context`, `change_history`, `decision_context`, `discussion_context` | none |
| **Q2_pr_rationale** | Rationale | Why PR #28479 was needed; main rationale for merging | 200 | (pattern matches Q1) | PR artifact + changes + decisions + discussions | none |
| **Q3_pr_risk** | Risk | Likely regressions and user-visible risks from PR #28479 | 200 | (pattern matches Q1) | semantic + artifact + decisions; change history available | none |
| **Q4_pr_related** | Related work | Other PRs/decisions closely related to PR #28479 | 200 | (pattern matches Q1) | semantic + decision + change history | none |
| **Q5_debug_scenario** | Debug plan | Debugging intermittent booking confirmation failures after #28479 | 200 | (pattern matches Q1) | semantic + artifact + decisions; partial structural context | none |

For **Q1_pr_summary** the layer reports:

```json
"coverage": {
  "status": "complete",
  "available": [
    "semantic_search",
    "artifact_context",
    "change_history",
    "decision_context",
    "discussion_context"
  ],
  "missing": []
}
```

Latency (Q1):

- `total_latency_ms ≈ 14616`
- `per_call_latency_ms`:
  - `semantic_search ≈ 3132`
  - `artifact_context ≈ 2875`
  - `change_history ≈ 2872`
  - `decision_context ≈ 2863`
  - `discussion_context ≈ 2874`

This is within budget but close to the 4s per-family cap.

---

## 2. Evidence for PR #28479

### 2.1 Artifact (PR-level summary)

From Q1:

- **Kind**: `pr`
- **Identifier**: `"28479"`
- **Title**: `fix: use i18n for apps count with proper pluralization`
- **Summary (excerpt)**:

  - Fixes **issue #28407** by implementing proper i18n pluralization for the apps count.
  - Branch: `fix/28407-i18n-apps-count`.
  - Commits:
    - `b7795bd…`: implement i18n pluralization and close #28407.
    - `517eb495…`: update e2e tests to match singular/plural behavior.
    - Two merge commits from `main` into the feature branch.

- **Author**: `Felipeness`

**Assessment**: Artifact context is rich and precise enough to answer:

- “What does PR #28479 do?”
- “Which issue does it fix?”
- “Which commits and authors were involved?”

### 2.2 Change history (PR-scoped)

Q1 includes a `changes` entry:

- **artifact_ref**: `PR #28479`
- **summary**:

  > Replaces hardcoded apps count string with i18n function for proper pluralization, fixes related tests, and updates localization keys.

- **pr_number**: `28479`
- **title**: as above
- **change_type**: `"fix"`
- **decisions**: embeds a key review comment from `Felipeness` explaining E2E failures and fixes.

**Assessment**: This is an accurate, high-level change summary tightly linked to the PR.

### 2.3 Decisions & discussions

The bundle exposes both **decision_context** and **discussion_context** for this PR.

Representative decision:

- **decision** (truncated):

  > Felipeness: Hey @sahitya-chandra, thanks for the heads up!
  >
  > I investigated the failures:
  >
  > - **E2E (1/8)**: was caused by my change — the test in `apps/web/playwright/fixtures/apps.ts` had a hardcoded `"1 apps"` assertion that now correctly renders as `"1 app"` (singular)… updated the test to handle singular/plural correctly…
  > - **E2E (2/8)** and **E2E API v2 (4/4)**: unrelated flaky failures…

- **pr_number**: `28479`

Matching discussion row:

- **source_ref**: `PR #28479 thread pr_conversation`
- **summary/headline/full_text**: same text as above, plus:

  - Explanation that **E2E (1/8)** failure is fixed by updating the test.
  - Confirmation that other failing shards are flaky and also failing elsewhere.
  - Final note that **CI is green** after the fix and PR is ready for re-review.

**Assessment**:

- The intelligence layer captures:
  - Root cause of a CI failure attributable to this PR.
  - A clear separation between **true regression** (changed expectation) and **flaky tests**.
  - Final “all green” signal.

This is ideal evidence for rationale/risk/debug questions.

### 2.4 Semantic context

Semantic hits include generic Graphiti facts:

- `FIXES` edges linking other PRs to their issues (e.g. #28622, #28647).
- `PART_OF_FEATURE` edges for related features.

While most examples in the snippet reference other PRs, this is still useful:

- To locate **other Cal.com PRs fixing related UI/i18n issues**.
- To answer “related work” queries (Q4).

---

## 3. Per-query evaluation

### Q1_pr_summary — “What and why”

> For Cal.com, summarize GitHub PR #28479: what does it change, and what problem is it trying to solve?

- **Evidence available**:
  - Artifact summary clearly states:
    - PR fixes apps count i18n pluralization.
    - Closes issue `#28407`.
    - Lists key commits and authors.
  - `changes` summary restates the change as:
    - Replace hardcoded count string with `t("number_apps", { count: installedAppsNumber })`.
    - Update tests and localization.

- **Expected ideal answer**:
  - Describe the bug: incorrect or non-i18n-safe apps count (e.g. “1 apps”).
  - Explain fix: switching to i18n pluralization API and updating tests.
  - Mention linkage to issue `#28407`.

- **Coverage**: `complete` across all PR-relevant families.

**Score (bundle quality)**: **3/3** — Everything needed for a clean, grounded summary is present.

### Q2_pr_rationale — “Why was it needed?”

> Why was PR #28479 needed in Cal.com? Explain the main rationale for merging it.

- **Evidence**:
  - Artifact summary ties PR to bug issue `#28407` and correctness of i18n pluralization.
  - Discussion shows the PR author acknowledging and fixing test failures due to the new behavior (`"1 app"` vs `"1 apps"`).

- **Expected ideal answer**:
  - State that the PR was merged to:
    - Correct user-facing pluralization of installed apps.
    - Align UI with i18n best practices.
    - Ensure tests reflect the correct behavior.

**Score (bundle quality)**: **3/3** — Clear rationale with direct evidence.

### Q3_pr_risk — “What could break?”

> For PR #28479 in Cal.com, what are the most likely regressions or user-visible risks introduced by this change?

- **Evidence**:
  - Discussions highlight that:
    - The PR directly affects apps count UI and related tests.
    - CI failures were limited to:
      - The test that asserted `"1 apps"` vs `"1 app"`.
      - Known flakiness elsewhere, not caused by this PR.
  - No structural evidence suggests it touches core booking logic or back-end flows.

- **Expected ideal answer**:
  - Narrow risk surface:
    - Misconfigured i18n keys or missing translations.
    - Any other UI components reusing the same key/string.
    - Test suites that still assume old strings.

**Score (bundle quality)**: **3/3** — Risk-relevant context is present and well-scoped.

### Q4_pr_related — “Related PRs and decisions”

> List other PRs and decisions that are closely related to PR #28479 in Cal.com (same files, feature area, or follow-up fixes).

- **Evidence**:
  - Semantic hits show:
    - Other `FIXES` and `PART_OF_FEATURE` events across Cal.com.
  - Decision and change-history families can be filtered by:
    - Same `pr_number`, or
    - Same file paths and feature labels once available.

- **Expected ideal answer**:
  - Name a handful of PRs that:
    - Touch the same UI area or i18n system.
    - Fix similar issues or extend the same feature.

**Score (bundle quality)**: **2.5/3** — Raw material is there; effectiveness depends on how well the agent filters and groups it.

### Q5_debug_scenario — “Debug plan after deployment”

> Suppose after deploying PR #28479, booking confirmations intermittently fail for recurring events. Using PR #28479 and nearby changes, outline a concrete debug plan.

- **Evidence**:
  - Bundle shows:
    - PR #28479 is localized to apps count UI and e2e tests.
    - No sign it touches booking confirmation or recurring event logic.

- **Expected ideal answer**:
  - Start from:
    - Apps count display.
    - i18n configuration and translation keys.
  - Then explicitly state:
    - “Given the scope of #28479, intermittent booking confirmation failures are likely unrelated.”
    - Suggest checking other PRs or code paths for booking flows.

**Score (bundle quality)**: **2.5/3** — Enough context to avoid over-blaming #28479, but robust debugging still requires combining this with broader code search.

---

## 4. Overall evaluation

### 4.1 Strengths

- **PR #28479 is very well represented**:
  - Complete artifact description.
  - Focused change summary.
  - Detailed reviewer discussion including:
    - Root cause analysis of CI failures.
    - Distinction between real regressions vs flaky tests.
    - Confirmation when CI is green.

- **Coverage is `complete`** for PR-centric families:
  - `semantic_search`
  - `artifact_context`
  - `change_history`
  - `decision_context`
  - `discussion_context`

### 4.2 Latency and performance

- Total time ≈ **14.6s** for a full bundle, with ~2.8–3.1s per family.
- Within the configured `timeout_ms=4000` per capability, but close enough that:
  - Timeouts could occur under load.
  - It’s important that the chat agent **does not re-call** graph tools once the bundle is prefetched.

### 4.3 Expected agent behavior (for QnA)

When `CONTEXT_INTELLIGENCE_ENABLED=true` and coverage is `complete` for PR #28479, the QnA agent should:

- **Answer directly from the prefetched block**:
  - Use artifact + changes for “what/why”.
  - Use decisions + discussions for rationale/risk.
  - Use semantic_hits + change_history for related work.

- **Avoid redundant tools**:
  - No extra `get_pr_review_context`, `get_decisions`, `get_change_history` calls for #28479 if the bundle is present and marked `complete`.

- **Use code tools sparingly**:
  - Only when the user explicitly asks for code/diffs (e.g., “show me the code changes in file X from PR #28479”).

---

## 5. Recommendations

1. **Keep PR #28479 as a “golden” PR test case** for:
   - Rationale, risk, and debug questions.
   - Testing whether the agent respects `coverage=complete` and avoids duplicate graph calls.

2. **Add a QnA-level test** mirroring the queries in this report:
   - Confirm the chat answers stay within the evidence shown here.
   - Confirm no unnecessary tool churn when the prefetch block is present.

3. **Monitor latency**:
   - If total bundle resolution gets slower under real load, consider:
     - Reducing families per query for non-PR questions.
     - Introducing caching for repeated PR queries in the same conversation.

---

## 6. Agent behavior on code-level queries (conversation API)

This section records **actual QnA agent responses** to code-level questions asked via:

- `POST /api/v1/conversations/{conversation_id}/message?stream=false`
- Conversation ID: `019d435a-0c4a-7813-93cc-d963e3f67627`

These questions are intentionally more code/flow centric than the earlier `resolve-context` tests.

### 6.1 CG1 — Files and recent change history

**Question (CG1)**  
> For Cal.com, focusing on PR #28479, which files did this PR modify for the apps count UI, and what is their recent change history?

**Observed agent answer (paraphrased, with key details):**

- The agent reports that context-graph PR tools like `get_pr_diff` / `get_change_history` are **currently erroring** with an internal error, so it cannot fetch fresh change history live.
- Based on previously retrieved PR details, it lists these files as modified by `#28479`:
  - `packages/platform/atoms/event-types/hooks/useTabsNavigations.tsx`  
    - UI change: replaces hardcoded `"${n} apps"` with i18n pluralization `t("number_apps", { count: installedAppsNumber })`.
  - `apps/web/playwright/fixtures/apps.ts`  
    - Test change: updates the Playwright helper to handle singular/plural (“app” vs “apps”) instead of asserting `"1 apps"`.
  - `apps/web/playwright/apps/analytics/analyticsApps.e2e.ts`  
    - Test change: updates calls to the helper to pass the correct counts.
- For **recent history**, it explicitly caveats that `get_change_history` is erroring now, but recalls from earlier successful calls that:
  - `useTabsNavigations.tsx` and `analyticsApps.e2e.ts` only showed **PR #28479** in the recent window.
  - `apps.ts` showed **PR #28479** plus a nearby follow‑up **PR #28486**.

**Assessment:**

- **Good**:
  - Names the correct files and describes the nature of each change.
  - Clearly surfaces the failure of `get_change_history` instead of hallucinating new history.
  - Reuses prior context-graph results (PR #28479 vs PR #28486) from memory.
- **Limitations**:
  - When the underlying context-graph tools error, it falls back to remembered history rather than e.g. live code search or git history.

**Score (agent behavior on CG1)**: **2.5/3** — Grounded and honest, but partially blocked by live tool errors.

---

### 6.2 CG2 — End-to-end code path and dependencies

**Question (CG2)**  
> In PR #28479 for cal.com, walk me through the end-to-end code path that turns the apps count data into the final string shown in the UI. Start from the data source where we compute the count, follow through any hooks/components/helpers, and end at the rendered text. Call out any key dependencies introduced or modified by this PR.

**Observed agent answer (key points):**

- **Data source**:
  - `apps/web/modules/event-types/components/EventTypeWebWrapper.tsx`
    - Uses `trpc.viewer.apps.integrations.useQuery(...)` to fetch `eventTypeApps`.
    - `eventTypeApps.items` contain installed integrations (`isInstalled`).
- **Count computation (core of PR #28479)**:
  - `packages/platform/atoms/event-types/hooks/useTabsNavigations.tsx`
    - `installedAppsNumber` = count of `eventTypeApps.items` where `isInstalled` is true.
    - `enabledAppsNumber` = count of `metadata.apps[slug]` entries that are enabled **and** correspond to installed apps.
- **String construction (the PR’s main change)**:
  - In `getNavigation()` inside `useTabsNavigations.tsx`:
    - The Apps tab’s `info` string becomes:  
      `info: \`${t("number_apps", { count: installedAppsNumber })}, ${enabledAppsNumber} ${t("active")}\``
    - This replaces a prior hardcoded `"${installedAppsNumber} apps"` string.
    - Relies on i18n pluralization for `number_apps`.
- **Propagation through layouts**:
  - `EventTypeWebWrapper` → `EventTypeComponent` → `EventTypeSingleLayout` / `EventTypeLayout`.
  - These pass `tabsNavigation` into shared navigation components.
- **Rendering to DOM**:
  - `packages/ui/components/navigation/tabs/VerticalTabs.tsx` + `VerticalTabItem.tsx`
    - `VerticalTabItem` renders:  
      `<p data-testid="apps-info" className="...">{info}</p>`
    - The “apps count” string is exactly the `info` built in `useTabsNavigations`.
- **Dependencies introduced/modified**:
  - No new NPM packages.
  - Stronger reliance on `number_apps` i18n pluralization.
  - Coupling to Playwright tests that assert the visible string:
    - `apps/web/playwright/fixtures/apps.ts`
    - `apps/web/playwright/apps/analytics/analyticsApps.e2e.ts`

**Assessment:**

- **Strengths**:
  - Correct, multi-hop call chain from tRPC to final DOM node.
  - Identifies the exact hook, layouts, and final component (`VerticalTabItem` with `data-testid="apps-info"`).
  - Connects UI code to tests and i18n keys.
- **Gaps**:
  - Answer is code-driven; doesn’t explicitly cite structural graph, but doesn’t pretend to.

**Score (agent behavior on CG2)**: **3/3** — Excellent end-to-end explanation with precise file and dependency mapping.

---

### 6.3 CG4 — Extending the apps-count UI safely

**Question (CG4)**  
> Imagine we need to extend the apps-count UI introduced in PR #28479 so that it also shows a separate count of apps that are installed but misconfigured. Based on the current code and tests, outline exactly which files and tests you would touch, and how you would thread this new count through the existing data flow without breaking current behavior.

**Observed agent answer (key points):**

- **Goal**: Change `"X apps, Y active"` into `"X apps, Y active, Z misconfigured"` while keeping behavior and tests stable.
- **Proposed file touch list**:
  - `packages/platform/atoms/event-types/hooks/useTabsNavigations.tsx`
    - Compute `misconfiguredAppsNumber` from either:
      - additional flags in `eventTypeApps.items` (e.g. `isMisconfigured`), or
      - extended `metadata.apps[slug]` fields (e.g. `configured` / `isValid`).
    - Thread the new count into the Apps tab `info` string, ideally **conditionally** when `misconfiguredAppsNumber > 0`.
  - `apps/web/modules/event-types/components/EventTypeWebWrapper.tsx`
    - If the current integrations query does not expose misconfiguration state, extend the tRPC procedure backing `viewer.apps.integrations` to include this.
  - i18n files under `packages/i18n/locales/**/common.json`
    - Add a new key for “misconfigured” or a plural `number_misconfigured_apps`, ensuring plural forms per locale.
  - E2E layer:
    - `apps/web/playwright/fixtures/apps.ts`
      - Update helpers that assert the apps info string to accept a new “misconfigured” argument, with a default of 0 to keep old test calls valid.
    - `apps/web/playwright/apps/analytics/analyticsApps.e2e.ts`
      - Update calls to the fixture helper where misconfigured apps should be asserted.
- **Safety / compatibility guidance**:
  - Do **not** change the semantics of `number_apps` (keep it purely “X app(s)”).
  - Append the misconfigured clause **conditionally** to avoid noisy UI and brittle tests.
  - Make Playwright assertions less brittle by using optional arguments or asserting substrings where reasonable.
  - Ensure the new “misconfigured” flag is deterministic in fixtures so tests don’t become flaky.

**Assessment:**

- Provides a concrete, file-level plan that:
  - Respects existing i18n design and avoids overloading `number_apps`.
  - Keeps E2E helpers backward compatible.
  - Respects current data boundaries (tRPC → wrapper → hook → UI).
- Explicitly calls out how to minimize breakage (defaults, conditional display).

**Score (agent behavior on CG4)**: **3/3** — High-quality, implementation-ready guidance consistent with the code structure.

---

### 6.4 CG3 — Dependency mapping via context graph (status)

**Question (CG3)**  
> Still in Cal.com PR #28479, show me which other components or features depend on the i18n key(s) and UI pieces that this PR touches. In particular, list any other files that use the same i18n key(s) or the same tabs/navigation components for apps, and explain how a future change to the apps-count string could impact them.

**Status:**

- The non‑streaming conversation call for CG3 is currently **hanging** in the agent/Celery pipeline (no HTTP response within ~3 minutes), despite the backend and Celery worker being healthy for other requests (CG1, CG2, CG4).
- This appears to be an issue in the live agent workflow for this specific query, not with `resolve-context` itself.

**Action item:** Once the Celery/agent hang is resolved, re-run CG3 and append the resulting answer here to complete the dependency‑mapping evaluation.
