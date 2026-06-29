# Context Engine - Tester Deploy-Readiness Flow

Use this as the human manual flow for public deploy readiness. It starts from a clean local Potpie home and tests the real product loop:

**setup -> integrations -> agent ingestion -> graph memory -> fresh-chat retrieval -> record/recall**

Primary harness: Cursor.

---

## 0. Start In The Repo

Open a fresh terminal in the repo you want to test.

```bash
export CONTEXT_ENGINE_HOME="/tmp/potpie-manual-e2e"
rm -rf "$CONTEXT_ENGINE_HOME"
mkdir -p "$CONTEXT_ENGINE_HOME"

pwd
git rev-parse --is-inside-work-tree
potpie --version
```

**Expected:** You are inside a git repo. `potpie --version` exits `0`. This run stores Potpie config, pots, and sources under `$CONTEXT_ENGINE_HOME`, not `~/.potpie`.

**What isolation means:**

| Layer | Isolated by `CONTEXT_ENGINE_HOME`? |
|---|---|
| Pots, sources, graph data, plans | Yes |
| GitHub/Linear/Jira login tokens | No — usually shared under `~/.config/potpie` |

So you are not mixing graph memory with daily `~/.potpie`, but integration logins may still be shared. That is expected.

Important: launch Cursor from this same terminal so the agent inherits the isolated home:

```bash
cursor .
```

If Cursor is already open from the Dock/app launcher, close it and reopen it from this terminal for this test.

---

## 1. Run Automated CE Smoke

Run this if you have the release test workbook:

```bash
python potpie/scripts/run_test_plan.py \
  --excel "/path/to/Tests.xlsx" \
  --repo "$(pwd)" \
  --no-checkout \
  --only-deterministic \
  --cli-cmd "potpie"
```

**Expected:** Deterministic Context Engine rows pass. If the workbook is unavailable, mark this as blocked and continue manual testing only with release-owner approval.

---

## 2. Setup Potpie And Cursor

Create one **named test pot** inside the isolated home (not your daily `~/.potpie` default):

```bash
potpie setup --repo . --agent cursor --pot qa-manual-e2e --yes --in-process
potpie skills install --agent cursor --scope project --path .
potpie status --host
potpie doctor
```

**Expected:**

- Setup exits `0`.
- Host/backend are ready or give a clear next action.
- Cursor skills/instructions are installed without deleting existing user text.
- Setup creates pot **`qa-manual-e2e`**, makes it active, and registers this repo to it.
- Config, pot metadata, and source metadata are under `$CONTEXT_ENGINE_HOME`.

Edge check:

```bash
potpie setup --repo . --agent cursor --backend falkordb_lite --yes --in-process
potpie doctor
```

**Expected:** Setup is idempotent. No duplicate/broken skill state.

---

## 3. Capture The Test Pot Id

Every step below (CLI, agent, UI) must use **this same pot**.

```bash
ls -la "$CONTEXT_ENGINE_HOME"
potpie --json pot info
potpie pot default show --repo current
potpie source list
potpie graph status --pot qa-manual-e2e
```

Copy the pot id from `pot info` (example: `pot_abc123`):

```bash
export POT_ID="<copied pot id>"
echo "Test pot: qa-manual-e2e ($POT_ID) home=$CONTEXT_ENGINE_HOME"
```

**Expected:**

- Active pot name is `qa-manual-e2e`.
- `$CONTEXT_ENGINE_HOME` contains `config.json` and `pots.json`.
- This repo is registered as a source on that pot.
- Graph is empty or near-empty before agent ingestion.

Use `--pot qa-manual-e2e` or `--pot "$POT_ID"` on commands when you want zero ambiguity.

---

## 4. GitHub Integration

```bash
potpie github login
potpie status --verify
```

Register the GitHub repo by replacing `owner/repo`:

```bash
potpie source add repo owner/repo
```

Pick one real PR or issue URL from the repo. You will paste it into the agent prompt later.

**Expected:** Login succeeds. `status --verify` passes. Private repo registration works if the account has access.

Manual edge checks:

1. Start `potpie github login` and cancel before completing auth.
2. Run `potpie status --verify`.
3. Complete `potpie github login` again.
4. Revoke or expire the GitHub credential from GitHub, then run:

```bash
potpie status --verify
potpie github login --force
potpie status --verify
```

**Expected:** Cancel is safe. Revoked auth is detected. Re-auth instructions are clear. No traceback.

---

## 5. Linear Integration

```bash
potpie linear login
potpie linear ls
```

From the `potpie linear ls` output, choose one real team key/id. It is usually a short value such as `ENG`, `CORE`, or `PLAT`. Write it down:

```text
LINEAR team key selected: ____________________
```

Then run:

```bash
potpie linear select --key "<paste selected Linear team key>" --limit 5
potpie status --verify
```

**Expected:** Login succeeds. Team/issues are readable.

Optional backend queue smoke, only if Linear queue ingest is in release scope:

```bash
potpie pot linear-team ingest "<paste selected Linear team key>" --count 50
```

**Expected:** Returns queued/applied/duplicate with an event id or batch id.

Manual edge check:

```bash
potpie linear logout
potpie linear ls
potpie linear login
```

**Expected:** Logged-out read fails with a clear login instruction. Re-login works.

---

## 6. Atlassian Integration: Jira And Confluence

Jira and Confluence use Atlassian credentials, but the CLI exposes them as separate product surfaces because Jira fetches projects/issues and Confluence fetches spaces/pages.

### Jira

```bash
potpie jira login
potpie jira ls
```

From the `potpie jira ls` output, choose one real project key. It is usually a short value such as `PROJ`, `CE`, or `ENG`. Write it down:

```text
Jira project key selected: ____________________
```

Then run:

```bash
potpie jira select --key "<paste selected Jira project key>" --limit 5
potpie status --verify
```

**Expected:** Jira login succeeds. Projects/issues are readable.

Optional backend queue smoke, only if Jira queue ingest is in release scope:

```bash
potpie pot jira-project ingest "<paste selected Jira project key>" --count 50
```

**Expected:** Returns queued/applied/duplicate with an event id or batch id.

Manual edge check:

```bash
potpie jira logout
potpie jira ls
potpie jira login --force
```

**Expected:** Logged-out read fails clearly. Forced login repairs auth.

### Confluence

```bash
potpie confluence login
potpie confluence ls
```

From the `potpie confluence ls` output, choose one real space key. It is usually a short value such as `DOCS`, `ENG`, or `RUNBOOKS`. Write it down:

```text
Confluence space key selected: ____________________
```

Then run:

```bash
potpie confluence select --key "<paste selected Confluence space key>" --limit 5
potpie status --verify
```

**Expected:** Confluence login succeeds. Spaces/pages are readable.

Manual edge check:

```bash
potpie confluence logout
potpie confluence ls
potpie confluence login --force
```

**Expected:** Logged-out read fails clearly. Forced login repairs auth.

---

## 7. Agent-Led Repo Ingestion

Open Cursor in this repo. Start a new agent chat and send:

```text
Use Potpie to ingest baseline memory for this repo into pot qa-manual-e2e (<POT_ID>).

Before writing anything, verify the isolated test environment:
- run `echo $CONTEXT_ENGINE_HOME`
- run `potpie --json pot info`
- run `potpie source list --pot qa-manual-e2e`

If `CONTEXT_ENGINE_HOME` is not /tmp/potpie-manual-e2e, stop and tell me Cursor was not launched with the isolated test environment.
If the active pot is not qa-manual-e2e (<POT_ID>), stop before writing.

Follow the Potpie source-ingestion and repo-baseline workflow:
- run pot/source/graph preflight
- inspect README, docs, manifests, entrypoints, configs, tests, and CI
- identify durable facts about repo purpose, features, services, dependencies, decisions, and preferences
- resolve entity identity before linking
- write evidence-backed graph memory through graph propose and graph commit --verify

Do not treat source add as ingestion. Do not invent facts from filenames alone.
Stop with what was committed and the evidence refs used.
```

Replace `<POT_ID>` in the prompt with the value from step 3.

**Expected:** Agent confirms `CONTEXT_ENGINE_HOME=/tmp/potpie-manual-e2e` and pot `qa-manual-e2e`, reads real files, runs graph preflight, commits graph memory, and summarizes evidence.

Verify:

```bash
potpie graph status --pot "$POT_ID"
potpie resolve "what does this repo do?" --pot "$POT_ID"
potpie search "$(basename "$(git rev-parse --show-toplevel)")" --pot "$POT_ID"
```

**Expected:** Results mention real repo facts. They must not mention synthetic `journey-service` test data.

---

## 8. Agent-Led Integration Ingestion

This is the main integration ingestion path for manual E2E:

**agent fetches source data using integration tools -> agent decides what is durable -> agent writes graph memory with evidence**

Do not use `potpie pot linear-team ingest` or `potpie pot jira-project ingest` as the agent's ingestion path. Those are only backend queue smoke checks.

Start a new Cursor chat and paste one prompt at a time.

### GitHub Prompt

```text
Use Potpie to ingest this GitHub PR/issue into pot qa-manual-e2e (<POT_ID>): <paste GitHub PR or issue URL>

Fetch the PR/issue content, comments, labels/status, linked files, and linked docs available through GitHub tools.
Record only durable recent-change, decision, bug/fix, feature, or integration facts with evidence refs.
Use graph propose and graph commit --verify. Put uncertain findings in graph inbox.
End with what was committed, what went to inbox, and what was skipped.
```

### Linear Prompt

```text
Use Potpie to ingest recent Linear context into pot qa-manual-e2e (<POT_ID>).

Use this Linear team key from the CLI test: <paste the selected Linear team key>.
Fetch a small set of recent issues/projects through Linear tools.
Record durable timeline, decision, bug/fix, or feature facts with evidence refs.
Use graph propose and graph commit --verify. Put uncertain findings in graph inbox.
```

### Jira Prompt

```text
Use Potpie to ingest recent Jira context into pot qa-manual-e2e (<POT_ID>).

Use this Jira project key from the CLI test: <paste the selected Jira project key>.
Fetch a small set of recent issues/status/changelog items through Jira tools.
Record durable timeline, decision, bug/fix, or feature facts with evidence refs.
Use graph propose and graph commit --verify. Put uncertain findings in graph inbox.
```

### Confluence Prompt

```text
Use Potpie to ingest Confluence context into pot qa-manual-e2e (<POT_ID>).

Use this Confluence space key from the CLI test: <paste the selected Confluence space key>.
Fetch a small set of relevant pages/runbooks/decisions through Confluence tools.
Record durable doc, runbook, decision, or workflow facts with evidence refs.
Use graph propose and graph commit --verify. Put uncertain findings in graph inbox.
```

**Expected:** Agent uses available integration tools or gives the exact missing-auth/tool blocker. Evidence refs name the source system and item.

Verify:

```bash
potpie resolve "what changed recently?"
potpie graph read --subgraph recent_changes --view timeline --limit 10 --format table
potpie graph inbox list
potpie graph history --limit 20
```

**Expected:** Timeline/history mention ingested integration items. Inbox contains only explicitly uncertain work.

---

## 9. Visual Graph Check

The UI is served by the **daemon**, which must use the same `$CONTEXT_ENGINE_HOME` as your terminal. Restart it in this shell, then open the UI for the test pot explicitly:

```bash
potpie daemon restart
potpie ui --pot qa-manual-e2e
```

**Expected:** Browser opens the graph explorer for pot `qa-manual-e2e`. You see the repo/integration facts the agent reported. If UI is empty but CLI reads work, the daemon was likely using a different home — rerun from the terminal where `CONTEXT_ENGINE_HOME` is set.

---

## 10. Fresh-Chat Retrieval

Start a fresh Cursor agent chat:

```text
Use Potpie context before answering.

Answer from project memory:
1. What does this repo do?
2. What important GitHub/Linear/Jira/Confluence context did we ingest?
3. Which files, services, features, docs, or tickets should I inspect first for related work?

Do not reread the whole repository unless Potpie context is missing.
```

**Expected:** Agent uses Potpie memory first, cites real paths/items/evidence, and says clearly if a memory area is sparse.

---

## 11. Record And Recall

In Cursor:

```text
Record this durable learning in Potpie:

Manual deploy-readiness E2E verified setup, GitHub, Linear, Jira, Confluence, Cursor agent ingestion, fresh-chat retrieval, and record/recall for this repo.

Use an appropriate record type and scope. Then verify it is searchable.
```

Verify:

```bash
potpie search "Manual deploy-readiness E2E verified"
```

**Expected:** Recorded learning is returned.

---

## 12. Other Harness Smoke

Run only for harnesses advertised in this release:

```bash
potpie skills install --agent claude --scope project --path .
potpie skills install --agent codex --scope project --path .
potpie skills install --agent opencode --scope project --path .
potpie skills status --agent cursor --scope project --path .
potpie skills status --agent claude --scope project --path .
potpie skills status --agent codex --scope project --path .
potpie skills status --agent opencode --scope project --path .
```

In each advertised harness, ask:

```text
Use Potpie project memory. What does this repo do, and what was recorded by the manual deploy-readiness E2E?
```

**Expected:** Skills install/status succeed. Existing user-authored instructions are preserved. Each harness retrieves memory or gives a clear setup/tooling blocker.

---

## 13. Final Checks

```bash
potpie status
potpie status --host
potpie doctor
```

**Expected:**

- `status` reports integration auth.
- `status --host` reports host/graph readiness.
- No output shows access tokens, API keys, private file contents, or unrelated PII.
- Docs/release notes say `source add` is registration-only, not ingestion.

---

## 14. Sign-Off

| Gate | Required? | Pass? |
|---|---:|---:|
| Automated deterministic CE tests | Yes | ☐ |
| Clean setup, pot, source, host | Yes | ☐ |
| GitHub happy path and auth edge cases | Yes | ☐ |
| Linear auth/read, plus queue smoke if in scope | If advertised | ☐ |
| Jira auth/read, plus queue smoke if in scope | If advertised | ☐ |
| Confluence auth/read | If advertised | ☐ |
| Cursor agent repo ingestion | Yes | ☐ |
| Agent-led integration ingestion | Yes for advertised integrations | ☐ |
| Potpie UI shows ingested memory | Yes | ☐ |
| Fresh-chat memory retrieval | Yes | ☐ |
| Record and recall | Yes | ☐ |
| Other harness smoke | If advertised | ☐ |
| Privacy/docs final checks | Yes | ☐ |

Block deploy if:

- `source add` is presented as ingestion.
- Agent writes graph facts without reading evidence.
- Fresh-chat retrieval cannot find committed memory.
- Integration auth failure lacks a clear next action.
- Wrong pot/repo memory is used silently.
- Tokens or secrets appear in output, logs, telemetry evidence, or agent response.

---

## Reset

```bash
potpie pot reset --confirm --pot qa-manual-e2e
rm -rf "$CONTEXT_ENGINE_HOME"
```

