# Implement Plan (Codex)

Use this command when executing a multi-step change. Follow the agreed plan, keep the user informed, and adapt responsibly as new information appears.

## Initial Response

- If a plan document or ticket path is provided, read it completely before acting. Confirm the objectives, scope, and acceptance criteria in your own words.
- If no plan exists, ask the user to share the desired outcome or reference material so you can build one together first.

## Workflow

### 1. Orient and confirm scope
- Summarize the plan back to the user and call out any ambiguities or missing decisions.
- If the work appears simple (≤10 minutes), confirm whether a lightweight approach is acceptable; otherwise create a multi-step plan and track it with the planning tool (`update_plan`) as you proceed.

### 2. Prepare to modify the code
- Locate the relevant files with `rg`, `ls`, or targeted reads. Review existing implementations to avoid regressions.
- Before editing, note expected side effects, dependencies, and tests that need updates.

### 3. Execute iteratively
- Implement changes in small, verifiable increments. Use `apply_patch` for manual edits when practical; prefer formatting and build tools only when necessary.
- After each significant change, update the plan status so progress stays transparent.
- Run relevant tests or commands from the repository root (or specified directory). Capture results briefly for the user; if a command cannot be run, explain why and suggest how they can verify locally.

### 4. Validate and polish
- Re-read modified sections to ensure consistency, coding standards, and accurate comments.
- Look for collateral updates (documentation, configs, migrations) that keep the system coherent.
- Summarize the diff mentally so you can explain the why and how for each file touched.

### 5. Wrap up
- Report which plan steps are complete, along with any remaining follow-ups or risks.
- Reference modified files with `path/to/file.ext:line` when describing the work.
- Note the tests you ran (or could not run) and any manual validation that remains.

Your goal is to land a clean, review-ready change set while keeping the user aware of trade-offs and outstanding tasks.
