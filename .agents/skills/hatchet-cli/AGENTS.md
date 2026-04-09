# Hatchet CLI Agent Skills

This skill package teaches AI agents how to use the Hatchet CLI to manage workflows, workers, and runs.

## When to use these skills

Read the relevant reference document before performing any Hatchet CLI task:

- **Setting up the CLI or creating a profile** → `references/setup-cli.md`
- **Starting a worker** → `references/start-worker.md`
- **Triggering a workflow and waiting for results** → `references/trigger-and-watch.md`
- **Debugging a failed or stuck run** → `references/debug-run.md`
- **Replaying a run with the same or new input** → `references/replay-run.md`

## Key conventions

- Always specify a profile with `-p HATCHET_PROFILE` unless a default profile is set.
- Use `-o json` for machine-readable output when parsing responses.
- Write workflow input to a temp file (e.g. `/tmp/hatchet-input-$(date +%s)-$$.json`) to avoid collisions.
- Clean up temp files after use.
