# Contributing to Potpie

Thank you for your interest in contributing to Potpie! We welcome and appreciate all contributions, whether you’re fixing bugs, improving documentation, or adding new features. This guide will help you get started.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [How to Contribute](#how-to-contribute)
3. [Getting Started](#getting-started)
4. [Development Workflow](#development-workflow)
5. [Submitting Pull Requests](#submitting-pull-requests)
6. [Documentation](#documentation)
7. [Community and Support](#community-and-support)

---

## Code of Conduct

Please note that by participating in the Potpie project, you agree to abide by our [Code of Conduct](./CODE_OF_CONDUCT.md). This is to ensure a positive, welcoming environment for all contributors.

## How to Contribute

There are several ways you can contribute to Potpie:

- Reporting bugs or suggesting improvements.
- Submitting feature requests.
- Adding documentation and improving existing content.
- Reviewing pull requests.
- Writing code and fixing issues.

If you’re unsure about how to get started, feel free to browse our issues labeled `good first issue` and `help wanted`. We also welcome you to join our discussions to understand the project better.

## Getting Started

### Prerequisites

Ensure you have the following software installed:

- [Git](https://git-scm.com/)
- [Python 3.12+](https://www.python.org/)
- [uv](https://docs.astral.sh/uv/) - Fast Python package installer (install via `curl -LsSf https://astral.sh/uv/install.sh | sh`)
- [Docker](https://www.docker.com/) (if applicable)

### Fork and Clone the Repository

1. **Fork** the repository on GitHub.
2. **Clone** your forked repository:

   ```bash
   git clone https://github.com/your-username/potpie.git
   cd potpie
   ```

3. Add the main repository as a remote:

   ```bash
   git remote add upstream https://github.com/potpie-ai/potpie.git
   ```

### Set Up Your Environment

Follow the project’s installation and setup instructions in the [README.md](./README.md) file to prepare your local environment.

## Development Workflow

To ensure consistency and quality in the codebase, please adhere to the following workflow:

1. **Create a Branch**:
   - Branches should be named according to the feature or bug fix. For example:

     ```bash
     git checkout -b feature/your-feature-name
     ```

2. **Make Changes**: Write clear, concise code and ensure that you follow any established coding conventions.

3. **Commit Your Changes**:
   - Write descriptive commit messages. Use imperative mood and provide context if necessary:

     ```bash
     git commit -m "Add feature to handle XYZ"
     ```

4. **Push to Your Fork**:

   ```bash
   git push origin feature/your-feature-name
   ```

## Testing

Tests have two owners: the root product and the context engine. The root suite
includes CLI, authentication, daemon, UI, documentation, parsing, sandbox, and
integration-package checks. The engine suite owns graph semantics, ingestion,
retrieval, persistence, and recovery.

```bash
make test-root
make test-context-engine
```

Each Python package runs in a separate process so its `tests` package cannot
shadow another suite. Parsing also runs its Rust tests; docs and UI use their
Node test runners. Focused package commands remain available:

```bash
make test-product TEST_CHECKS=python
make test-parsing
make test-sandbox
make test-integrations
make test-context-engine TEST_TIER=graph
make test-context-engine TEST_TIER=restart
```

Use shared fixtures for stable interfaces and named parameter rows for input,
provider, platform, or backend variations. Keep recovery scenarios explicit when
they require ordered actions and intermediate assertions. Source-layout and
packaging assertions use the `source_contract` marker; runtime behavior belongs
in tests that can run against installed distributions.

CI builds every distribution before test jobs begin: product and workspace
wheels and source archives, native parsing wheels, and the repaired Linux ARM64
FalkorDB Lite wheel. A manifest records the commit, package version, wheel tags,
and checksum. Smoke jobs install those exact artifacts in clean environments
outside the checkout. Root and engine wheels are installed together when testing
the product, and the engine is also installed independently.

Deeper suites run on selected targets, with separate graph-acceptance and serial
restart profiles. Required native acceptance cases fail if skipped. Source
contracts run separately from installed runtime tests so source imports cannot
mask missing files or dependency errors in a wheel.

## Submitting Pull Requests

When you’re ready to submit your changes:

1. **Open a Pull Request (PR)**:
   - Go to the Potpie repository and select **New Pull Request**.
   - Choose your branch and provide a meaningful title and description.

2. **Describe Your Changes**:
   - Include a detailed description of what you have done and why.
   - Link related issues (e.g., `Closes #123`).

3. **Request a Review**:
   - After submitting your PR, request a review from a project maintainer.

4. **Respond to Feedback**:
   - Be responsive to any feedback you receive. Once changes are approved, a maintainer will merge your PR.

## Documentation

User-facing docs live under [`docs/`](../docs/). If your change affects a command, setup step, or integration that users see, update the matching page there.

## Community and Support

Open a Github issue to connect with other contributors, ask questions, and get support.

Thank you for your contribution to Potpie! 🥧
