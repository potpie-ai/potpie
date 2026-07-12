# Potpie Skills Architecture

> Verified at `f435fb4` on 2026-07-13. Skills are a root product capability
> governed by [PKG-SKILL-001](../../spec/modules/package-boundary.md).

Potpie skills teach a coding harness how to use the Potpie CLI and four-tool MCP
surface. They are product resources, not graph facts, engine operations, or
server-side agents.

## Ownership and flow

```text
potpie/skills/resources
  → SkillCatalog
  → SkillService
  → target adapter
  → harness global or project directory
```

Root `potpie` owns:

- the catalog and bundled templates;
- target detection and filesystem layout;
- install, update, remove, add, list, status, and drift behavior;
- managed instruction blocks in `AGENTS.md` and `CLAUDE.md`;
- the static CLI command manifest and snippet validation.

The context engine has no skill manager, template bundle, target knowledge, or
Typer dependency.

## Packaged resources

Resources live under `potpie/skills/resources/templates/` and ship in the root
wheel:

```text
templates/
├── agent_bundle/          AGENTS.md and .agents/skills for Codex/Cursor/OpenCode
├── claude_bundle/         CLAUDE.md, commands, and Claude skills
├── claude_plugin/         Claude plugin metadata, hooks, commands, and skills
└── global_agent_bundle/   compact managed global instruction blocks
```

The engine wheel contains only engine-owned reconciliation playbooks used by
optional engine adapters; those are not installed into user harnesses.

## Commands

```text
potpie skills list
potpie skills install [ID] --agent <agent> [--scope global|project] [--path PATH]
potpie skills update [ID] [--all] --agent <agent>
potpie skills remove [ID] [--all] --agent <agent>
potpie skills status --agent <agent>
potpie skills add PATH
```

`setup` calls the same root service. CLI command handlers do not inspect Typer
or copy package files themselves.

## Installation targets

| Harness | Default global skill directory |
|---|---|
| Codex | `$HOME/.agents/skills/<skill>/` |
| Claude Code | `~/.claude/skills/<skill>/` |
| Cursor | `~/.cursor/skills/<skill>/` |
| OpenCode | `~/.config/opencode/skills/<skill>/` |

Project scope writes below the explicit `--path`. Target adapters are
root-owned because directory conventions, harness detection, prompts, and
installation policy are product behavior.

## Managed instruction blocks

Where a harness uses a global or repository instruction file, Potpie owns only
the text between:

```text
<!-- potpie-start -->
...
<!-- potpie-end -->
```

Install/update replaces that block in place or appends it if absent. Remove
deletes the block. Content outside the markers is preserved byte-for-byte.

## Static command validation

Runtime installation does not import or introspect Typer. Instead:

1. `scripts/generate_skill_command_manifest.py` walks the registered command
   tree during development/build verification.
2. `potpie/skills/command_manifest.json` records the approved workflow-first
   command paths.
3. Tests compare the manifest with live Typer registration.
4. Every bundled Markdown/Python/JSON command reference is checked against the
   manifest.
5. Wheel-based lifecycle tests install resources from an artifact, not the
   source checkout.

This catches stale commands such as top-level provider groups, `graph mutate`,
or `potpie use` before release without coupling installation to CLI internals.

## Skill surface

Skills may instruct agents to call:

- `potpie resolve`, `search`, `record`, and `status`;
- workflow-first pot/source/timeline commands;
- graph catalog/read/search and propose/commit workflows;
- integration status and provider commands;
- the four MCP tools when an MCP client is available.

They must not invent a fifth MCP tool, use a removed alias, bypass
propose/commit for writes, or depend on private engine imports.

## Status and drift

`SkillService.status()` reports installed, missing, and outdated entries.
Product status reduces this to `ready`, `missing`, `outdated`, or `unknown` and
may recommend `potpie skills update --all`. The pure engine status report does
not contain skill state.

## Adding or changing a bundled skill

1. Change the canonical resource under `potpie/skills/resources/templates`.
2. Use only commands in the static manifest.
3. Add/update catalog metadata and target expectations.
4. Run resource, snippet, manifest, and installed-wheel lifecycle tests.
5. If the CLI contract must change, update the boundary spec and command
   manifest first; do not add an installation-time alias.
