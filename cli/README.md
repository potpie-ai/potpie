# Potpie CLI

Command-line interface for local Potpie usage.

This CLI provides a command-line interface for interacting with Potpie locally.

## Commands

- `potpie start` - Start the Potpie server
- `potpie stop` - Stop the Potpie server  
- `potpie parse <repo-path> [--branch <branch-name>]` - Parse a repository
- `potpie chat <project-id> --agent <agent-name> [--branch <branch-name>]` - Chat with an agent
- `potpie status <project-id>` - Check parsing status of a project

## Installation

```bash
pip install -e .
```

## Environment

Set `POTPIE_DIR` environment variable to point to your Potpie installation directory.
If not set, defaults to current working directory.

## Requirements

- Python 3.11+
- Potpie server installation
- Docker (for running the Potpie server)
