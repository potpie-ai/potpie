# Contributing to Potpie

Contribute bug fixes, features, documentation, and code reviews. This guide covers
local setup and the pull request workflow.

## Table of contents

1. [Code of conduct](#code-of-conduct)
2. [How to contribute](#how-to-contribute)
3. [Getting started](#getting-started)
4. [Development workflow](#development-workflow)
5. [Testing](#testing)
6. [Submitting pull requests](#submitting-pull-requests)
7. [Documentation](#documentation)
8. [Community and support](#community-and-support)

## Code of conduct

Participation in the Potpie project requires following our
[code of conduct](./CODE_OF_CONDUCT.md).

## How to contribute

- Report bugs or request features.
- Add or improve documentation.
- Review pull requests.
- Write code and fix issues.

Look for issues labeled `good first issue` or `help wanted` if you need a place
to start.

## Getting started

### Prerequisites

Install the following:

- [Git](https://git-scm.com/)
- [Python 3.12+](https://www.python.org/)
- [uv](https://docs.astral.sh/uv/), the Python package installer
- [Docker](https://www.docker.com/), if your development setup requires it

### Fork and clone the repository

1. Fork the repository on GitHub.
2. Clone your fork:

   ```bash
   git clone https://github.com/your-username/potpie.git
   cd potpie
   ```

3. Add the main repository as a remote:

   ```bash
   git remote add upstream https://github.com/potpie-ai/potpie.git
   ```

### Set up your environment

Follow the installation and setup instructions in the [README](../README.md).

## Development workflow

1. Create a branch named for the feature or bug fix:

   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and follow the existing code conventions.
3. Commit with a message that states what the change does:

   ```bash
   git commit -m "Add feature to handle XYZ"
   ```

4. Push to your fork:

   ```bash
   git push origin feature/your-feature-name
   ```

## Testing

Write unit tests for new features and update tests when behavior changes. Run the
relevant tests before opening a pull request.

## Submitting pull requests

1. Open a pull request against the Potpie repository with a descriptive title.
2. Explain what changed and why. Link related issues, such as `Closes #123`.
3. Request a review from a project maintainer and address the feedback.

A maintainer merges the pull request after approval.

## Documentation

User-facing docs live under [`docs/`](../docs/). If your change affects a command,
setup step, or integration that users see, update the matching page there.

## Community and support

Open a [GitHub issue](https://github.com/potpie-ai/potpie/issues) to ask questions,
report bugs, or request changes.
