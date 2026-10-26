<p align="center">
  <a href="https://potpie.ai?utm_source=github">
    <img src="https://raw.githubusercontent.com/potpie-ai/potpie/main/assets/readme_logo_light.svg" alt="Potpie AI logo" />
  </a>
</p>

<p align="center">
  <a href="https://docs.potpie.ai"><img src="https://img.shields.io/badge/Docs-potpie.ai-111827?style=for-the-badge&logo=readthedocs&logoColor=white&labelColor=22c55e" alt="Docs"></a>
  <a href="https://github.com/potpie-ai/potpie/actions/workflows/test.yml"><img src="https://img.shields.io/github/actions/workflow/status/potpie-ai/potpie/test.yml?branch=main&style=for-the-badge&label=Tests&logo=githubactions&logoColor=white&labelColor=111827" alt="Tests"></a>
  <a href="https://pypi.org/project/potpie/"><img src="https://img.shields.io/pypi/v/potpie?style=for-the-badge&label=PyPI&logo=pypi&logoColor=white&labelColor=111827&color=3775A9" alt="PyPI"></a>
  <a href="https://discord.gg/ryk5CMD5v6"><img src="https://img.shields.io/badge/Discord-Join-5865F2?style=for-the-badge&logo=discord&logoColor=white&labelColor=111827" alt="Discord"></a>
  <a href="https://github.com/potpie-ai/potpie/blob/main/LICENSE"><img src="https://img.shields.io/github/license/potpie-ai/potpie?style=for-the-badge&label=License&color=64748b&labelColor=111827" alt="Apache 2.0"></a>
</p>

[Potpie](https://potpie.ai) turns your codebase and software development lifecycle into a living context graph for AI agents.
It indexes code, structure, decisions, source history, team knowledge and engineering workflows, so agents can answer questions, plan changes, debug failures, and write code with project-specific context.

![Potpie context graph demo](https://raw.githubusercontent.com/potpie-ai/potpie/main/assets/context_graph.gif)

## Install and setup Potpie

### Step 1: install the CLI through [PyPI](https://pypi.org/project/potpie/) with `uv` or `pip`:

```bash
uv tool install potpie
```

or:

```bash
python3 -m pip install --user potpie
```

> [!NOTE]
> `uv tool install potpie` is recommended for CLI installs because global
> mutation of Python packages is generally not recommended.

### Step 2: Run the Potpie setup wizard

```bash
potpie setup
```

The setup wizard provisions local config, storage, the daemon, a default pot, and
agent skills. It also lets you choose integrations and the coding harness Potpie
should configure.

![Potpie setup wizard](https://raw.githubusercontent.com/potpie-ai/potpie/main/assets/wiz_screen.png)

### Step 3: Open your configured harness

Potpie is already integrated into your selected harness.
You can start using Potpie with the repo of your choice.

Open your previously selected harness and ask it to use Potpie for the repo.
![Potpie in OpenAI Codex](https://raw.githubusercontent.com/potpie-ai/potpie/main/assets/codex_potpie.png)


> [!NOTE]
> You don't need to run a separate manual ingest command. The CLI registers
> sources and the configured agent can ingest or update project context when the
> task requires it.

You can view your context graph in the web UI:

```bash
potpie ui
```

This will open a graph explorer in your browser.
![Potpie web UI](https://raw.githubusercontent.com/potpie-ai/potpie/main/assets/web_ui.png)

## Basic CLI user checklist

The main CLI commands are:

| Command | Purpose |
| --- | --- |
| `potpie setup` | Run first-time local setup for config, daemon, default pot, and agent skills. |
| `potpie login` | Sign in to Potpie for account-backed and managed features. |
| `potpie github login` / `potpie linear login` | Connect source integrations you want agents to use. |
| `potpie status` | Show context readiness for the active pot, including daemon, graph, and skill checks. |
| `potpie auth status` | Show configured integration auth status. |
| `potpie auth status --verify` | Verify integration credentials with lightweight API checks. |
| `potpie doctor` | Run local diagnostics for daemon, backend capabilities, and skill drift. |
| `potpie source add repo .` | Register the current repo as a source for the resolved pot. |
| `potpie pot list` / `potpie pot use <id-or-name>` | List pots and choose the active workspace. |
| `potpie resolve "<task>"` | Pull the context an agent should read before doing a task. |
| `potpie search "<query>"` | Look up a specific file, workflow, bug, decision, or convention. |
| `potpie record --type <type> --summary "<summary>"` | Write a durable project learning. |
| `potpie graph ...` | Use lower-level graph reads, quality checks, proposals, and commits. |
| `potpie ui` | Open the local graph explorer served by the daemon. |
| `potpie skills install --agent <agent>` | Install or refresh Potpie guidance for an agent harness. |

Examples:

```bash
potpie setup --repo . --agent claude
potpie github login
potpie auth status
potpie status
potpie source add repo .
potpie resolve "what should I know before working in this repository?"
potpie search "authentication flow"
potpie record --type decision --summary "Prefer the Potpie CLI for graph work"
```

You can find an exhaustive list with more examples in our [docs](https://docs.potpie.ai).

## Integrations and Coding Harnesses

Potpie supports a variety of integrations and coding harnesses, with more coming.
If your team needs a new integration or harness, please
[raise a ticket](https://github.com/potpie-ai/potpie/issues/new/choose).

### Integrations

| Tool | Description |
| --- | --- |
| ![GitHub][github-badge] | Index repositories, pull requests, issues, reviews and source history. |
| ![Linear][linear-badge] | Index teams, issues, projects and documents. |
| ![Jira][jira-badge] | Index projects, issues, status and changelog context. |
| ![Confluence][confluence-badge] | Index spaces, pages, runbooks and decisions. |

### Coding harnesses

| Tool | Description |
| --- | --- |
| ![Claude Code][claude-badge] | Install Potpie instructions and skills for Claude Code. |
| ![OpenAI Codex][codex-badge] | Install Potpie instructions and skills for OpenAI Codex. |
| ![Cursor][cursor-badge] | Install Potpie instructions and skills for Cursor. |
| ![OpenCode][opencode-badge] | Install Potpie skills for OpenCode. |

[github-badge]: https://img.shields.io/badge/GitHub-111827?style=flat-square&logo=github&logoColor=white
[linear-badge]: https://img.shields.io/badge/Linear-111827?style=flat-square&logo=linear&logoColor=white
[jira-badge]: https://img.shields.io/badge/Jira-111827?style=flat-square&logo=jira&logoColor=white
[confluence-badge]: https://img.shields.io/badge/Confluence-111827?style=flat-square&logo=confluence&logoColor=white
[claude-badge]: https://img.shields.io/badge/Claude%20Code-111827?style=flat-square&logo=anthropic&logoColor=white
[codex-badge]: https://img.shields.io/badge/OpenAI%20Codex-111827?style=flat-square&logo=openai&logoColor=white
[cursor-badge]: https://img.shields.io/badge/Cursor-111827?style=flat-square&logo=cursor&logoColor=white
[opencode-badge]: https://img.shields.io/badge/OpenCode-111827?style=flat-square&logo=opencode&logoColor=white

## Architecture

Potpie's current architecture is CLI-first. CLI is designed to be used by both humans and agents. Read the deeper architecture notes in [`docs/context-graph/architecture.md`](https://github.com/potpie-ai/potpie/blob/main/docs/context-graph/architecture.md).

## License

This project is licensed under the Apache 2.0 License. See [LICENSE](https://github.com/potpie-ai/potpie/blob/main/LICENSE).


## Community & Support

- [GitHub Issues](https://github.com/potpie-ai/potpie/issues): bugs and repository-scoped requests
- [Discord](https://discord.gg/ryk5CMD5v6): community discussion and support
- [Docs](https://docs.potpie.ai): setup, product guides, and integration details

New contributions are always welcome. Read the [Contributing Guide](https://github.com/potpie-ai/potpie/blob/main/.github/CONTRIBUTING.md)
to set up your environment, understand the workflow, and open a pull request.

<a href="https://github.com/potpie-ai/potpie/graphs/contributors">
  <img src="https://contributors-img.web.app/image?repo=potpie-ai/potpie" alt="Contributors" />
</a>
