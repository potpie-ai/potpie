# Implementation Plan (Codex)

Use this command when you need to partner with the user on a detailed implementation plan. Stay skeptical, verify everything in the codebase, and surface open questions early.

## Initial Response

- If the invocation includes a ticket path, spec, or file list, read each file completely right away using `shell` commands (`cat`, `sed`, `rg`). Summarize the salient requirements before asking follow-up questions.
- If no context is provided, reply with:
```
I'll help you create a detailed implementation plan. Please share:
1. The task or ticket description (or a file path to it)
2. Constraints, acceptance criteria, or dependencies
3. Any prior research or related changes I should review

Once I have that context we can iterate on the plan.
```
Then wait for the user's response.

## Workflow

### 1. Collect context
- Read every referenced document in full; avoid partial reads or skipping sections.
- Capture assumptions, risks, blockers, and unknowns while reading.
- Note any missing information that must be confirmed with the user.

### 2. Investigate the codebase
- Map requirements to actual code. Use `rg`, `ls`, and targeted file reads to identify relevant modules, APIs, and tests.
- Skim enough implementation detail to understand data flow, entry points, and side effects.
- Record important findings with `path/to/file.ext:line` references.

### 3. Synthesize understanding
- Restate the problem in your own words, grounded in what you saw in the repo.
- Highlight current behavior, gaps, and technical constraints that will influence the solution.
- Flag contradictions or uncertainties that need clarification before planning further.

### 4. Draft the plan
- Organize work into logical phases or milestones that someone else could follow.
- For each phase:
  - List concrete engineering tasks (code edits, migrations, configuration changes, tests, rollouts).
  - Call out the files or systems likely touched.
  - Note risks, mitigations, and validation steps.
- Include supporting work (documentation, communication, feature flags) when relevant.

### 5. Validate coverage
- Ensure the plan addresses every stated requirement, edge case, and dependency.
- Explicitly list any open questions or decisions awaiting input.
- Recommend follow-up research if something still feels uncertain.

## Output Style

- Begin with a short overview paragraph summarizing goal, approach, and key risks.
- Follow with numbered phases containing bullet task lists.
- Reference files with `path/to/file.ext:line` when possible.
- Close with open questions, follow-up actions, and suggested validation steps.
