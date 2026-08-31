---
title: First steps
description: Run the Potpie setup wizard, open your coding harness, and view the context graph.
---

## Prerequisites

Complete [Installation](./installation.md).

## Step 2: Run the Potpie setup wizard

```bash
potpie setup
```

The setup wizard provisions local config, storage, the daemon, a default pot, and
agent skills. It also lets you choose integrations and the coding harness Potpie
should configure.

![Potpie setup wizard](https://raw.githubusercontent.com/potpie-ai/potpie/main/assets/wiz_screen.png)

## Step 3: Open your configured harness

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
