# Potpie Cursor hooks

Installed by `potpie install --agent cursor`. Requires `potpie` on PATH (or
`POTPIE_BIN`) and an active pot (`potpie pot use` or `POTPIE_POT`).

| Event | Effect |
|---|---|
| `beforeSubmitPrompt` | Remember prompt (`lineage capture --remember-prompt`) |
| `afterFileEdit` | Link edited span to latest prompt (fail-open) |
| `sessionStart` / `preToolUse` / `postToolUse` / `stop` | Graph nudge inject (same adapter as Claude) |

Enable hooks in Cursor **Settings → Hooks**. Set `POTPIE_HOOK_DEBUG=1` to log
adapter decisions to stderr.
