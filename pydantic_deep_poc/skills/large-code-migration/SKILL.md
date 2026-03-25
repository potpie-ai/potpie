---
name: large-code-migration
description: Playbook for bounded discovery, explicit slicing, single-writer implementation, and final verification.
---

# Large Code Migration

Use this skill for migrations that touch multiple files, frameworks, or runtime paths.

## Core workflow

1. Do a bounded discovery pass.
2. Identify the smallest set of impacted files and symbols.
3. Split the work into independent implementation slices.
4. Parallelize only read-only discovery or verification slices.
5. Run implementation slices serially in a single shared CCM session.
6. Run a single verification and rollup pass at the end.

## Rules

- The orchestrator does not write code directly.
- Implementation workers get context packets and operate only within those packets.
- Avoid sending workers to rediscover facts already gathered.
- Keep external docs short and extract only the specific APIs or startup patterns needed.

## Delegation packet

Include:

- `TASK_TYPE`
- `OBJECTIVE`
- `FILES_IN_SCOPE`
- `KNOWN_FINDINGS`
- `CONSTRAINTS`
- `DONE_WHEN`

## Parallelization guidance

Parallelize only for read-only work in this PoC.

Good examples:

- discovery across separate module families
- docs lookup plus local file inspection
- independent verification checks

Bad examples:

- two workers editing any staged file set
- concurrent changes to the same shared settings file
- multiple implementers writing into the same CCM session

## Completion

The final verifier should report:

- validated commands
- files covered
- remaining risks
- whether the diff is ready to apply
