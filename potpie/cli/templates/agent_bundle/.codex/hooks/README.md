# Potpie Codex hooks

Installed by `potpie install --agent codex`. Same adapter as Cursor/Claude;
uses `--harness codex` for session keys.

Requires `potpie` on PATH and an active pot. Wire hooks per your Codex harness
docs if the event names differ; the adapter accepts `--event user_prompt`,
`post_edit`, and direct nudge events.
