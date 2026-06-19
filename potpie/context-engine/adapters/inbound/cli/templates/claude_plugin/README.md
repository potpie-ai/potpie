# Potpie plugin for Claude Code

Wires Claude Code lifecycle events to Potpie's project-memory graph so relevant
context surfaces **without the agent being asked**, and durable learnings can be
captured at the right moments. The hook path is **model-free**: it forwards event
shape to `potpie graph nudge` and injects the returned context/instruction. The
reasoning - deciding what's true, which entity, and how to phrase it - happens in
your session on your subscription.

## What it does

| Claude event (hook hint) | Resolves to nudge | Nudge | Effect |
|---|---|---|---|
| `SessionStart` → `session_start` | `session_start` | inject | repo baseline: active decisions + repo-level preferences |
| `PreToolUse(Write\|Edit)` → `pre_edit` | `pre_edit` | inject | preferences + known bug-patterns scoped to the file being edited |
| `PreToolUse(Bash)` → `bash_pre` | `pre_deploy` *(deploy/infra command only; else silent)* | inject | env-qualified service neighborhood for a deploy command |
| `PostToolUse(Bash)` → `bash_post` | `test_failed` *(test command, failed)* | inject | prior-occurrence matches by symptom + recent changes for scope |
| `PostToolUse(Bash)` → `bash_post` | `test_passed` *(test command, passed)* | instruct | "you resolved X after editing Y — record the bug+fix if non-obvious" |
| `Stop` → `stop` | `stop` | instruct | "capture durable learnings (new prefs, decisions, fixes)" |

`hooks.json` wires each event to a coarse hint (`session_start`, `pre_edit`,
`bash_pre`, `bash_post`, `stop`). The adapter (`hooks/potpie_nudge.py`) performs
mechanical refinement: a `bash_pre` becomes `pre_deploy` only for a deploy/infra
command, a `bash_post` becomes `test_failed`/`test_passed` only for a test command
and by its outcome, and anything else stays silent. It then forwards exactly one
`potpie graph nudge` call and injects the result. This refinement makes **no model
call**. The adapter is fail-safe: any error or a missing `potpie` binary means it
injects nothing and exits cleanly, so a hook problem can never block your session.

## Requirements

- The `potpie` CLI on `PATH` (or set `POTPIE_BIN`). `python3` on `PATH`.
- An active pot for the project (`potpie pot use <id>`), or set `POTPIE_POT`.

## Install

**Via marketplace (recommended):**

```
/plugin marketplace add /path/to/this/plugin/dir
/plugin install potpie@potpie
```

**Repo-local hooks (no marketplace):** add the hook commands to your project's
`.claude/settings.json` under `hooks`, pointing at this directory's
`hooks/potpie_nudge.py` and replacing `${CLAUDE_PLUGIN_ROOT}` with the directory
path. See `hooks/hooks.json` for the exact event→command mapping.

`potpie install --agent claude-plugin` drops this whole directory into your repo.

## Skills

The plugin bundles the graph contract skill plus use-case workflow skills:
preferences, infra architecture, change timeline, debug memory, and harness-led
source ingestion. Skills run in-session on your subscription.

## Other harnesses

The same adapter works for Codex and Cursor via `--harness codex|cursor`; their hook
systems differ, so wire them per their docs to invoke
`python3 potpie_nudge.py --harness <name> --event <hint>`. The adapter's payload
accessors already tolerate common key shapes (`session_id`, `tool_input.file_path`,
`tool_input.command`).

## Debugging

Set `POTPIE_HOOK_DEBUG=1` to log adapter decisions to stderr (visible in transcript
mode). `POTPIE_HOOK_TIMEOUT` (seconds, default 15) bounds the nudge subprocess.
