# Codex-Informed Retrofit Plan for the `pydantic-deep` PoC

## Summary
Keep `pydantic-deep` as the base, but retrofit the PoC around Codex-style control patterns: scoped `AGENTS.md` for repo rules, role-specific prompt layers, parent-mediated subagent communication, depth-limited delegation, and tool-level parallel safety rather than blanket parallelism. The benchmark task stays broad by design; the system should become better at handling arbitrary large migrations, not learn a special-purpose `celery-to-hatchet` skill. Codex itself separates repo-scoped `AGENTS.md` from reusable skills, applies role configs at spawn time, routes delegated user-input requests back through the parent, and gates real parallel execution per tool.

The current PoC pressure points are concrete: `factory.py` still gives the supervisor too much power, `prompts_loader.py` is carrying too much behavior in a single prompt, `toolsets_builder.py` still exposes shell too broadly, and `deps.py` still shares mutable run state across subagents. Use these as the primary integration points:

- `/Users/deepesh/work/potpie/pydantic_deep_poc/poc/agents/factory.py`
- `/Users/deepesh/work/potpie/pydantic_deep_poc/poc/agents/prompts_loader.py`
- `/Users/deepesh/work/potpie/pydantic_deep_poc/poc/tools/toolsets_builder.py`
- `/Users/deepesh/work/potpie/pydantic_deep_poc/poc/managers/deps.py`

## Implementation Changes

### 1. Prompt and instruction architecture
- Replace the single long supervisor prompt with four layers:
  - **Base agent prompt**: short Codex-style operating rules only.
  - **Repo policy layer**: root `AGENTS.md` for Potpie-wide invariants.
  - **Role prompt layer**: one prompt file each for `orchestrator`, `discover`, `implement`, `verify`.
  - **Skill layer**: loaded only for reusable workflows like bounded discovery, CCM-only edits, or verification.
- Put all repo invariants into `AGENTS.md`, not the main prompt.
  - Include: “only harness creates worktrees”, “only implementer may stage code”, “no raw git branch/worktree commands”, “success requires verifier pass”, “summary must be derived from diff”.
- Use skills only for general reusable capabilities, not benchmark-specific migrations.
  - Create `bounded-discovery`
  - Create `ccm-only-implementation`
  - Create `repo-verifier`
  - Optionally create `large-code-migration`
- Do **not** create a `celery-to-hatchet` skill.
  - The benchmark should stay broad and adversarial.
  - The system must generalize to other migrations.
- Add a small prompt loader registry so roles are loaded from files rather than embedded strings.
  - `prompts/roles/orchestrator.md`
  - `prompts/roles/discover.md`
  - `prompts/roles/implement.md`
  - `prompts/roles/verify.md`
- Keep role prompts short and hard-edged.
  - Explicit budgets, allowed tools, blocked behaviors, output schema, stop conditions.
- Mirror Codex’s role treatment: role files override behavior at spawn time without silently changing the parent’s provider/model defaults unless explicitly configured.

### 2. Agent roles and tool boundaries
- Introduce an internal `RoleSpec` config object with:
  - `name`
  - `prompt_path`
  - `toolset_ids`
  - `can_delegate`
  - `can_write_ccm`
  - `can_apply`
  - `can_use_shell`
  - `shell_mode` = `none | read_only | validate_only`
  - `may_run_async`
  - `may_ask_parent`
- Enforce this role matrix:
  - **Orchestrator**: may delegate, may manage todos, may inspect summaries, may call `show_diff` and `apply_changes`, no shell, no file edits, no direct repo exploration.
  - **Discover**: read-only file tools, docs/web, read-only shell, async allowed, no CCM, no git mutation.
  - **Implement**: CCM write tools plus limited validate-only shell, sync only, no branch/worktree/git mutation, no docs/web.
  - **Verify**: read-only file tools, CCM summary tools, `show_diff`, validate-only shell, no CCM writes, no apply.
- Remove these from every role except the harness:
  - `checkout_worktree_branch`
  - any branch creation tool
  - `git add`
  - `git checkout`
  - `git reset`
  - shell-based file creation or in-place editing
- Keep shell, but split it into policy-enforced modes:
  - `read_only_shell`: `rg`, `find`, `ls`, `git diff`, `git status`, `sed -n`, `cat`, `wc`
  - `validate_only_shell`: `python -m py_compile`, targeted tests, import checks, `git diff --stat`
  - reject heredocs, redirections, `sed -i`, `perl -pi`, `mv`, `cp`, `rm`, `git add`, `git checkout`, `git commit`
- Add a command retry guard keyed by `(role, normalized_command, normalized_error_signature)`.
  - One retry max for the same failure.
  - Second identical failure forces `BLOCKED` or strategy change.
- Do not expose `apply_changes` to workers.
  - Only orchestrator may call it, and only after verifier returns `pass=true`.

### 3. Subagent lifecycle, communication, and state isolation
- Replace free-form subagent outputs with a strict `WorkerResult` schema:
  - `status`: `completed | blocked | failed`
  - `summary`
  - `files_in_scope`
  - `files_touched`
  - `facts`
  - `checks_run`
  - `policy_violations`
  - `question` (only for `blocked`)
- Make delegation packets typed and mandatory:
  - `TASK_TYPE`
  - `OBJECTIVE`
  - `FILES_IN_SCOPE`
  - `KNOWN_FINDINGS`
  - `CONSTRAINTS`
  - `DONE_WHEN`
  - `OUT_OF_SCOPE`
- Enforce a single-writer design:
  - only one `implement` worker may exist at a time
  - discovery and verification may run in parallel
  - no parallel implementation until per-worker staging exists
- Replace shared mutable `RunContext` cloning with `ScopedRunContext`:
  - `discover`: no CCM access
  - `verify`: read-only CCM access
  - `implement`: write CCM access
  - `orchestrator`: summary/apply access only
- Add runtime assertions so a role cannot call a disallowed tool even if the model tries.
- Adopt Codex’s parent-mediated clarification pattern:
  - any worker that may need clarification must run async
  - orchestrator handles `WAITING_FOR_ANSWER` via `check_task` / `answer_subagent`
  - sync workers are not allowed to call `ask_parent`; they must return `status=blocked` with one question
- Add subagent depth control.
  - `orchestrator -> worker` only
  - worker spawning is disabled
  - if depth limit is hit, delegation features are turned off, matching Codex’s behavior of disabling collaboration features once `agent_max_depth` is exceeded.
- Surface active subagents in environment context for traceability, similar to Codex’s environment-context subagent listing.

### 4. Orchestrator control loop
- Replace the current “explore until ready” loop with a fixed 5-stage state machine:
  1. `bootstrap`
  2. `bounded_discovery`
  3. `plan_slices`
  4. `implement_serial`
  5. `verify_and_finalize`
- Hard budgets:
  - orchestrator discovery: max 6 tool calls or 2 turns
  - each discovery worker: max 6 tool calls
  - implement worker: no broad exploration; only files in scope plus directly imported dependencies
  - verifier: max 8 tool calls
- Orchestrator behavior:
  - load matching general skill first if applicable
  - dispatch 1–3 read-only discovery workers in parallel
  - merge findings into a deterministic plan
  - run implementation slices serially
  - run exactly one verifier pass
  - if verifier fails, either one repair pass or return failure
- Add contradiction detection:
  - if discovery recorded “existing implementation found”, later prompts cannot claim it is missing without a fresh cited file check
  - if verifier data contradicts the narrative summary, finalization is blocked
- Add deterministic final summary generation:
  - source material only from `WorkerResult`, `git diff --name-only`, `git diff --stat`, verifier checks
  - no free-form “migration complete” language unless verifier sets `all_acceptance_criteria_passed=true`
- Codex’s orchestrator template encourages subagents and parallel coordination, but in this PoC that should apply only to read-only work until writer isolation exists.

### 5. Verification and completion gates
- Introduce a required `VerificationReport` with:
  - `ccm_files_count > 0`
  - `show_diff_non_empty = true`
  - `apply_ready = true`
  - `todos_clear = true`
  - `policy_violations = []`
  - `objective_checks = [...]`
  - `pass = true | false`
- For any large migration benchmark, the verifier must check:
  - key legacy-surface metrics decreased relative to baseline
  - no duplicate parallel implementation stack was invented unless explicitly allowed
  - targeted runtime paths no longer depend on the legacy system where they were supposed to be migrated
  - touched files compile where applicable
  - any generated summary or migration note matches actual changed files
- `apply_changes` preconditions:
  - verifier pass is true
  - staged file count > 0
  - no pending or in-progress todos
  - current worktree path equals harness worktree path
- Treat these as hard failures, not warnings:
  - empty CCM on finalization
  - worker wrote outside CCM
  - worker created or switched worktree/branch
  - repeated identical shell failure
  - final summary claims completion while verifier failed
- Add a guardian-like review hook for risky future actions.
  - Not for v1 implementation writes.
  - Only for sandbox escape, network escalation, or external side effects.
  - This mirrors Codex’s “guardian reviewer” direction without expanding scope into your core migration loop.

## Public Interfaces and Types
- Add `RoleSpec` and `ToolPolicy` internal types for role assembly in `factory.py`.
- Add `ScopedRunContext` or `RunContextCapabilities` to replace raw shared `poc_run` mutation in `deps.py`.
- Add `WorkerResult`, `DelegationPacket`, and `VerificationReport` Pydantic models.
- Change subagent prompt contracts so every worker returns structured JSON/text in the same schema.
- Add a prompt/skill registry interface:
  - `load_role_prompt(role_name)`
  - `select_skills(task_text, repo_context)`
  - `render_repo_policy(context_path)`

## Test Plan
- Unit tests:
  - role-to-tool matrix prevents disallowed calls
  - `ScopedRunContext` blocks CCM writes outside implementer
  - retry guard stops repeated identical shell failures
  - sync worker cannot use parent-question path
  - async worker question path reaches orchestrator answer flow
- Integration tests:
  - clean benchmark creates only the harness worktree
  - orchestrator delegates within budget
  - discovery workers run in parallel and implementer stays single-writer
  - `show_diff` empty blocks finalization
  - `apply_changes` cannot run before verifier pass
- Trace assertions:
  - at least 1 `task()` call in complex tasks
  - 0 shell-based file writes
  - 0 branch/worktree creation by workers
  - 0 duplicate failing shell commands beyond retry budget
  - final completion only when `VerificationReport.pass=true`
- Acceptance benchmark:
  - rerun `task_celery_to_hatchet`
  - compare against baseline on token count, tool count, delegation count, repeated-command count, verification success

## Assumptions and Defaults
- Keep `pydantic-deep` as the runtime base for v1.
- Keep `moonshotai/kimi-k2.5` as the benchmark model, but treat model parallelism as opportunistic; enforce safe parallelism in the orchestrator regardless of model behavior.
- Restrict parallelism to read-only discovery and verification until per-worker isolated staging exists.
- Keep shell available, but only under role-specific read/validate policies.
- Use `AGENTS.md` for scoped repo invariants and skills for reusable workflows; do not duplicate the same rules in both places. Codex uses `AGENTS.md` as scoped instruction files and skills as discoverable bundles of instructions, scripts, and resources.
