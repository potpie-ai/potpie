---
name: sandbox-recovery
description: How to use the cloned-repo sandbox tools and recover from their failure modes (clone-in-progress, unknown ref, truncation, ambiguous/unknown repo, transient unavailability). Load this when a sandbox_* tool errors or returns an unexpected shape.
version: "1.0.0"
tags: [sandbox, recovery, ingestion]
---

# Sandbox tools and their failure modes

The pot may have one or more repositories cloned into a shared sandbox. The
`sandbox_*` tools let you read source, walk git history, and switch refs to
ground the graph in real code instead of webhook summaries. Use them to
**inspect, not mutate**.

## Usage discipline

- Call `sandbox_list_repos()` **first**. If more than one repo is attached,
  every other sandbox tool requires `repo='owner/name'`; with a single repo the
  argument is optional.
- Prefer the dedicated read tools (`sandbox_read_file`, `sandbox_list_dir`,
  `sandbox_search`) for plain reads. Reach for `sandbox_exec` only when you need
  shell pipes or flags they don't expose.
- For ANY text/code search use ripgrep (`rg -n PATTERN`, `rg -t py PATTERN`,
  `rg -l PATTERN | head`) — never `grep -r` or `find … -exec grep`; `rg` is
  preinstalled, far faster, and honours `.gitignore`. `fd`, `jq`, `tree`,
  `git`, `python3`, and `node` are also available.
- A long-running `sandbox_exec` returns `running=true` with a `session_id`;
  drive it with `sandbox_write_stdin` and poll until `running=false` to capture
  the `exit_code`. Always kill a session you no longer need so it doesn't hold
  the sandbox.

## Budget

40 sandbox calls for `repository.added`; 15 for other events. Stop walking once
you have enough to write the plan — the loop is not a free exploration session.

## Failure-mode ladder

- **Clone in progress.** The sandbox is provisioned lazily. If
  `sandbox_list_repos` returns an empty list, or `sandbox_read_file` errors with
  a clone-related message, retry once after another tool call. If still empty,
  fall back to graph + GitHub tools and add a warning — do not fabricate data.
- **`sandbox_checkout` → `{error, kind}`.**
  - `kind='unknown_ref'`: the ref doesn't exist — warn, do NOT invent the commit.
  - `kind='conflict'`: the worktree has uncommitted state; retry with
    `force=True` only when the event payload requires that specific ref.
  - `kind='network'` / `'auth'`: transient/configuration — warn and continue
    with graph tools.
- **`truncated: true`** on a large file/diff. Narrow the path list, or use
  `sandbox_git_log` + `sandbox_git_show` to walk commits one at a time.
- **`{"error": "ambiguous_repo"}`**: you omitted `repo=` on a multi-repo pot —
  re-issue with one of the repos listed under `available`.
- **`{"error": "unknown_repo"}`**: the repo isn't attached to this pot. Don't
  guess; call `sandbox_list_repos` again and use a returned name.
- **`{"kind": "sandbox_unavailable", "transient": true}`**: the infrastructure
  (Daytona / snapshot pull) failed before the call reached the worktree. Retry
  once; if it still fails, skip the sandbox for this batch and continue with
  graph + GitHub tools rather than aborting. Do NOT fabricate file contents.

The rule under every branch: when the sandbox can't give you a fact, record a
warning and degrade to the graph + connector tools — never invent the contents
the sandbox would have returned.
