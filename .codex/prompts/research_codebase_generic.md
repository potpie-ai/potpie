# Research Codebase (Codex)

Invoke this command when the user needs a deep understanding of how something works today. Your job is to investigate the repository, surface relevant code, and explain behavior with evidence.

## Initial Response

- If the invocation names a feature area, file, or ticket, restate the exact research goal and confirm any constraints (time period, stack slice, environment).
- If context is missing, ask the user to clarify what they want to learn and why, so you can focus the investigation.

## Workflow

### 1. Frame the questions
- Translate the user's request into concrete questions you can answer with code or configuration evidence.
- Identify key data flows, services, and edge cases that must be inspected.

### 2. Locate relevant artifacts
- Use `rg`, `ls`, and targeted `find`/`fd` commands to discover source files, tests, migrations, configs, and docs.
- Follow the call chain: trace entry points, handlers, models, background jobs, and integrations as needed.
- Read the important files fully; avoid quoting snippets out of context.

### 3. Analyze and corroborate
- Explain what the code is doing, why, and under which conditions. Link related pieces together (controllers ↔ services ↔ DB, etc.).
- Capture important details with `path/to/file.ext:line` references so the user can jump into the code quickly.
- Note inconsistencies, TODOs, feature flags, or tech debt that might affect future changes.

### 4. Summarize findings
- Present results in a structured narrative:
  - Current behavior and data flow
  - Key components and responsibilities
  - Known edge cases, failure modes, or constraints
  - Open questions or areas needing confirmation from humans or production data
- Highlight reusable patterns or prior implementations that could inform upcoming work.

## Output Style

- Stay concise but thorough—favor facts grounded in the code over speculation.
- Use bullet lists for related findings and short paragraphs for nuanced explanations.
- Call out next steps or suggested follow-up investigations when appropriate.
