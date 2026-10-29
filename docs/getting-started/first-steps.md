---
title: First steps
description: Run the Potpie setup wizard, open your coding harness, and view the context graph.
---

## Prerequisites

Complete [Installation](./installation.md).

## Step 2: run the Potpie setup wizard

```bash
potpie setup
```

The setup wizard creates local config and storage, starts the daemon, and sets
up a default pot and agent skills. Choose the integrations and coding harness
Potpie should configure.

![Potpie setup wizard](https://raw.githubusercontent.com/potpie-ai/potpie/main/assets/wiz_screen.png)

## Step 3: open your configured harness

Open the harness you selected during setup and ask it to use Potpie for your repo.
![Potpie in OpenAI Codex](https://raw.githubusercontent.com/potpie-ai/potpie/main/assets/codex_potpie.png)

> [!NOTE]
> You don't need to run a separate manual ingest command. The CLI registers
> sources and the configured agent can ingest or update project context when the
> task requires it.

You can view your context graph in the web UI:

```bash
potpie ui
```

The command opens a graph explorer in your browser.
![Potpie web UI](https://raw.githubusercontent.com/potpie-ai/potpie/main/assets/web_ui.png)
