# VS Code Extension Debug/DAP Handoff

Date: 2026-05-25

## Context

The debug agent can reach the local workspace over the Socket.IO tunnel, but C/C++
debugging currently fails or degrades in two ways:

1. Terminal commands such as `gcc -g ...` and `mkdir ...` can complete locally while
   the backend waits until the RPC timeout because no `tool_response` is observed.
2. `get_workspace_debug_context` reports `debug_adapters: ["python", "node"]` even
   on a macOS machine where LLDB and `lldb-dap` are installed. This blocks C/C++
   debug session setup from the agent's point of view.

This is not expected to be a local network or WebSocket reliability problem. The
backend timeout means: `tool_call` was emitted, but no matching `tool_response`
arrived on Redis pub/sub before the timeout.

Latest isolated harness verification after the extension-side terminal fix:

- `/api/terminal/execute` for `cc -g -o add_numbers add_numbers.c` returned in
  roughly 400ms with `success=true`, `exit_code=0`, empty stdout/stderr, and
  `timed_out=false`.
- `/api/workspace/debug-context` returned C/C++-relevant adapters:
  `["python", "node", "lldb", "lldb-dap"]`.
- `/api/debug/start-session` with `lldb-dap` succeeded and returned a session id.
- `/api/debug/set-breakpoints` at `add_numbers.c:4` returned a verified
  breakpoint.
- `/api/debug/snapshot` after continue still failed with
  `Cannot take snapshot: session is running, expected paused`.

That harness run was a smoke test, not a full DAP tool matrix. It did not cover
`step-over`, `step-into`, `step-out`, `evaluate`, `list-sessions`, or a clean
`stop-session` path.

## Backend Contract

Socket event from backend to extension:

```json
{
  "correlation_id": "<uuid>",
  "endpoint": "/api/terminal/execute",
  "payload": {},
  "timeout": 35.0
}
```

The extension must always respond with:

```json
{
  "correlation_id": "<same uuid>",
  "success": true,
  "result": {}
}
```

or:

```json
{
  "correlation_id": "<same uuid>",
  "success": false,
  "error": "..."
}
```

Important backend routes used by the debug agent:

| Endpoint | Purpose |
| --- | --- |
| `/api/workspace/debug-context` | Return launch configs, available debug adapters, inferred commands |
| `/api/terminal/execute` | Run shell commands in sync or async mode |
| `/api/terminal/sessions/{session_id}/output` | Poll async terminal output |
| `/api/terminal/sessions/{session_id}/signal` | Stop async terminal process |
| `/api/debug/start-session` | Start DAP debug session |
| `/api/debug/set-breakpoints` | Set breakpoints |
| `/api/debug/snapshot` | Capture stack/locals/expressions |
| `/api/debug/list-sessions` | List debug sessions |
| `/api/debug/stop-session` | Stop debug session |

## Required Extension Changes

### 1. Always Emit `tool_response`

Every extension route handler must emit a response for every received
`correlation_id`, including:

- success with empty stdout
- non-zero exit
- thrown exception
- timeout
- unknown route
- debug adapter launch failure

This should be implemented with a top-level `try/catch/finally` around dispatch so
no handler can accidentally leave the backend waiting.

### 2. Fix Terminal Completion For Silent Commands

The terminal executor must not infer completion from visible terminal output.
Commands like these are valid and often produce no stdout:

```bash
gcc -g -o add_numbers add_numbers.c
mkdir -p .vscode
true
```

The user-facing contract is that terminal commands should be visible in a VS Code
integrated terminal by default. A hidden `child_process` result is acceptable only
for an explicitly hidden/internal mode; it is not acceptable for normal debug-agent
terminal commands because users must be able to see what the agent ran.

Recommended behavior:

- Open or reuse a named integrated terminal such as `Potpie`.
- Show the command in that terminal before execution.
- Capture stdout, stderr, exit code, duration, and timeout for the RPC response.
- Append an internal sentinel to detect completion reliably when the command is
  silent.
- Keep the terminal visible unless the payload explicitly requests hidden
  execution.

Implementation options:

- If using an integrated terminal/PTY, append an explicit sentinel that includes
  exit code, then parse that sentinel, e.g. `printf "\n__POTPIE_EXIT:$?__\n"`.
- If using `child_process.spawn` / `execFile`, wire it only to an explicit hidden
  mode or mirror the command/output into the visible terminal.

Expected sync result shape:

```json
{
  "success": true,
  "command": "gcc -g -o add_numbers add_numbers.c",
  "output": "",
  "error": "",
  "exit_code": 0,
  "duration_ms": 1234
}
```

### 3. Return Accurate `debug_adapters`

`debug_adapters` should represent debug adapter types that the extension can
actually launch in the current VS Code/Cursor session.

For C/C++ on macOS, include at least one of:

- `lldb` when CodeLLDB (`vadimcn.vscode-lldb`) is available
- `cppdbg` when Microsoft C/C++ (`ms-vscode.cpptools`) is available
- `lldb-dap` or another explicit adapter id only if the extension start-session
  handler can launch it directly

Do not infer this only from binaries on disk. `/usr/bin/lldb` and
`/Library/Developer/CommandLineTools/usr/bin/lldb-dap` prove the machine has LLDB,
but they do not prove VS Code has a registered adapter type unless the extension can
use them.

If the extension can launch `/Library/Developer/CommandLineTools/usr/bin/lldb-dap`
directly without a marketplace extension, return a capability that makes that
explicit, for example:

```json
{
  "debug_adapters": ["python", "node", "lldb-dap"],
  "native_debuggers": {
    "lldb": "/usr/bin/lldb",
    "lldb_dap": "/Library/Developer/CommandLineTools/usr/bin/lldb-dap"
  }
}
```

### 4. Support C/C++ Launch Configs

The extension should accept launch configs from `.vscode/launch.json` and start
sessions for C/C++ configs such as:

```json
{
  "type": "cppdbg",
  "request": "launch",
  "name": "Debug add_numbers",
  "program": "/Users/deepesh/work/valkey/add_numbers",
  "cwd": "/Users/deepesh/work/valkey",
  "MIMode": "lldb"
}
```

and/or CodeLLDB style:

```json
{
  "type": "lldb",
  "request": "launch",
  "name": "Debug add_numbers",
  "program": "/Users/deepesh/work/valkey/add_numbers",
  "cwd": "/Users/deepesh/work/valkey"
}
```

If an adapter is missing, return a structured failure response instead of timing
out:

```json
{
  "success": false,
  "error": "debug_adapter_unavailable",
  "message": "No registered debug adapter for type 'cppdbg'. Install ms-vscode.cpptools or use lldb."
}
```

### 5. Add Correlation-ID Tracing

Log these lifecycle points with `correlation_id`, `endpoint`, and `workspace_id`:

- received `tool_call`
- selected handler
- handler started
- handler completed
- emitted `tool_response`
- handler exception
- handler timeout

This is the fastest way to distinguish:

- backend emitted to stale socket id
- extension received but handler did not respond
- extension responded but socket server did not publish to Redis
- backend worker did not receive Redis pub/sub message

## Acceptance Checks

With the extension connected to the workspace:

1. `execute_terminal_command("true")` returns exit code 0 without timeout.
2. `execute_terminal_command("mkdir -p .vscode")` returns exit code 0 without timeout.
3. `execute_terminal_command("gcc -g -o add_numbers add_numbers.c")` returns exit
   code 0 without timeout, even with empty stdout.
4. `get_workspace_debug_context` includes the C/C++ adapter that the extension can
   actually launch (`lldb`, `cppdbg`, or `lldb-dap`).
5. `start_debug_session` for the compiled `add_numbers` binary returns a session id
   or a structured adapter-unavailable error; it must not time out silently.
6. `set_breakpoints` at `add_numbers.c:4` and `take_debug_snapshot` return locals
   when the program pauses.
7. User-facing terminal commands visibly open or reuse an integrated terminal and
   show the command being run; the RPC response must not be produced only by a
   hidden process in normal mode.

## Backend Follow-Up Needed

The backend currently models `start_debug_session.language` as:

```python
Literal["python", "node", "go", "unknown"]
```

To make C/C++ first-class, backend should add `c`, `cpp`, or a direct launch-config
type field and pass through adapter-specific launch configs without forcing them
through language-only routing.
