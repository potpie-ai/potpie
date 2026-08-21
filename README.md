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

## Documentation

User-facing docs live under [`docs/`](./docs/) and follow the Hub and Spoke layout.

| Page | What it covers |
| --- | --- |
| [Installation](./docs/getting-started/installation.md) | Install the CLI from PyPI with `uv` or `pip` |
| [First steps](./docs/getting-started/first-steps.md) | Setup wizard, coding harness, and `potpie ui` |
| [Integrations and coding harnesses](./docs/guides/integrations.md) | GitHub, Linear, Jira, Confluence, and supported harnesses |
| [CLI reference](./docs/reference/cli.md) | Main commands and examples |

The published portal is [docs.potpie.ai](https://docs.potpie.ai).

Validate the Spoke docs tree locally:

```bash
node scripts/validate-docs-cli.mjs docs potpie
node --test scripts/validate-pr.test.mjs
```

## Install

```bash
uv tool install potpie
```

or:

```bash
python3 -m pip install --user potpie
```

Then run `potpie setup`. Full steps are in [Installation](./docs/getting-started/installation.md) and [First steps](./docs/getting-started/first-steps.md).

## Architecture

Potpie's current architecture is CLI-first. CLI is designed to be used by both humans and agents. Read the deeper architecture notes in [`internal-docs/context-graph/architecture.md`](./internal-docs/context-graph/architecture.md).

## License

This project is licensed under the Apache 2.0 License. See [LICENSE](./LICENSE).

## Community & Support

- [GitHub Issues](https://github.com/potpie-ai/potpie/issues): bugs and repository-scoped requests
- [Discord](https://discord.gg/ryk5CMD5v6): community discussion and support
- [Docs](https://docs.potpie.ai): setup, product guides, and integration details

New contributions are always welcome. Read the [Contributing Guide](https://github.com/potpie-ai/potpie/blob/main/.github/CONTRIBUTING.md)
to set up your environment, understand the workflow, and open a pull request.

<a href="https://github.com/potpie-ai/potpie/graphs/contributors">
  <img src="https://contributors-img.web.app/image?repo=potpie-ai/potpie" alt="Contributors" />
</a>
