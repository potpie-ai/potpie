
<p align="center">
  <a href="https://potpie.ai?utm_source=github">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="./assets/readme_logo_dark.svg" />
      <source media="(prefers-color-scheme: light)" srcset="./assets/logo_light.svg" />
      <img src="./assets/logo_light.svg"  alt="Potpie AI logo" />
    </picture>
  </a>
</p>


# Potpie

[Potpie](https://potpie.ai) is an open-source platform that creates AI agents specialized in your codebase. Build a comprehensive knowledge graph of your code and let agents handle everything from debugging to feature development.


<p align="center">
<img width="700" alt="Potpie Dashboard" src="./assets/home_page.png" />
</p>

<p align="center">
  <a href="https://docs.potpie.ai"><img src="https://img.shields.io/badge/Docs-Read-blue?logo=readthedocs&logoColor=white" alt="Docs"></a>
  <a href="https://github.com/potpie-ai/potpie/blob/main/LICENSE"><img src="https://img.shields.io/github/license/potpie-ai/potpie" alt="Apache 2.0"></a>
  <a href="https://github.com/potpie-ai/potpie"><img src="https://img.shields.io/github/stars/potpie-ai/potpie" alt="GitHub Stars"></a>
  <a href="https://discord.gg/ryk5CMD5v6"><img src="https://img.shields.io/badge/Discord-Join-5865F2?logo=discord&logoColor=white" alt="Discord"></a>
  <a href="https://marketplace.visualstudio.com/items?itemName=PotpieAI.potpie-vscode-extension"><img src="https://custom-icon-badges.demolab.com/badge/VSCode-Extension-0078d7.svg?logo=vsc&logoColor=white" alt="VSCode Extension"></a>
</p>


## Quick Start

### Prerequisites

- [Docker](https://docker.com) installed and running
- [Git](https://git-scm.com) installed
- [Python 3.11+](https://python.org) with [uv](https://docs.astral.sh/uv/)

### Installation

1. **Clone the repository**

   ```bash
   git clone --recurse-submodules https://github.com/potpie-ai/potpie.git
   cd potpie
   ```

2. **Configure your environment**

   ```bash
   cp .env.template .env
   ```

   Edit `.env` with the following required values:

   ```bash
   # App & Environment
   isDevelopmentMode=enabled
   ENV=development
   defaultUsername=defaultuser

   # AI / LLM Configuration
   LLM_PROVIDER=openai                    # openai | ollama | anthropic | openrouter
   OPENAI_API_KEY=sk-proj-your-key
   CHAT_MODEL=gpt-4o
   INFERENCE_MODEL=gpt-4o-mini

   # Database
   POSTGRES_SERVER=postgresql://postgres:mysecretpassword@localhost:5432/momentum
   NEO4J_URI=bolt://127.0.0.1:7687
   NEO4J_USERNAME=neo4j
   NEO4J_PASSWORD=mysecretpassword

   # Redis & Background Jobs
   REDISHOST=127.0.0.1
   REDISPORT=6379
   BROKER_URL=redis://127.0.0.1:6379/0
   CELERY_QUEUE_NAME=dev

   # Project & Repo Management
   PROJECT_PATH=projects
   ```

   > **`CHAT_MODEL`** and **`INFERENCE_MODEL`** are used for agent reasoning and knowledge graph generation respectively. Model names follow the `provider/model_name` format as expected by [LiteLLM](https://docs.litellm.ai/docs/providers).

   > **💡 Using Ollama instead?** Set `LLM_PROVIDER=ollama` and use `CHAT_MODEL=ollama_chat/qwen2.5-coder:7b` and `INFERENCE_MODEL=ollama_chat/qwen2.5-coder:7b`.

   See `.env.template` for the full list of optional configuration (logging, feature flags, object storage, email, analytics, etc.).

3. **Install dependencies**

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   uv sync
   ```

4. **Start all services**

   ```bash
   chmod +x scripts/start.sh
   ./scripts/start.sh
   ```

   This will start Docker services, apply migrations, start the FastAPI app, and start the Celery worker.

5. **Health Check**

   ```bash
   curl -X GET 'http://localhost:8001/health'
   ```

6. **Check parsing status**

   ```bash
   curl -X GET 'http://localhost:8001/api/v1/parsing-status/your-project-id'
   ```

To stop all services:

```bash
./scripts/stop.sh
```

#### Now set up Potpie.ai Frontend

```bash
cd potpie-ui

cp .env.template .env

pnpm build && pnpm start
```


## GitHub Authentication

| Method | Configuration | Best For |
|--------|--------------|----------|
| **GitHub App** | `GITHUB_APP_ID`, `GITHUB_PRIVATE_KEY` | Production |
| **PAT Pool** | `GH_TOKEN_LIST=ghp_token1,ghp_token2` | Development / Higher rate limits |
| **Unauthenticated** | No configuration required | Public repositories only (60 req/hr) |

Set `GITHUB_AUTH_MODE` to `app`, `pat`, or `none` to select the method.

---

## Self-Hosted Git Providers

For self-hosted Git servers (e.g., GitBucket, GitLab, etc.), configure:

```bash
CODE_PROVIDER=github         # github | gitlab | gitbucket
CODE_PROVIDER_BASE_URL=http://your-git-server.com/api/v3
CODE_PROVIDER_TOKEN=your-token
```

## 🤖 Potpie's Prebuilt Agents

Potpie offers a suite of specialized codebase agents for automating and optimizing key aspects of software development:

- **Debugging Agent**: Automatically analyzes stacktraces and provides debugging steps specific to your codebase. [Docs](https://docs.potpie.ai/agents/debugging-agent)
- **Codebase Q&A Agent**: Answers questions about your codebase and explains functions, features, and architecture. [Docs](https://docs.potpie.ai/agents/qna-agent)
- **Code Changes Agent**: Analyzes code changes, identifies affected APIs, and suggests improvements before merging. [Docs](https://docs.potpie.ai/agents/code-changes-agent)
- **Integration Test Agent**: Generates integration test plans and code for flows to ensure components work together properly. [Docs](https://docs.potpie.ai/agents/integration-test-agent)
- **Unit Test Agent**: Automatically creates unit test plan and code for individual functions to enhance test coverage. [Docs](https://docs.potpie.ai/agents/unit-test-agent)
- **LLD Agent**: Creates a low level design for implementing a new feature by providing functional requirements to this agent. [Docs](https://docs.potpie.ai/agents/introduction)
- **Code Generation Agent**: Generates code for new features, refactors existing code, and suggests optimizations. [Docs](https://docs.potpie.ai/agents/introduction)

### Custom Agents

With Custom Agents, you can design personalized tools that handle repeatable tasks with precision. Define:

- **System Instructions** — The agent's task, goal, and expected output
- **Tasks** — Individual steps for job completion
- **Tools** — Functions for querying the knowledge graph or retrieving code

```bash
curl -X POST "http://localhost:8001/api/v1/custom-agents/agents/auto" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "An agent that takes stacktrace as input and gives root cause analysis and proposed solution as output"}'
```

Read more in our [documentation](https://docs.potpie.ai/open-source/agents/create-agent-from-prompt).

## Use Cases

- **Onboarding** : Help developers new to a codebase understand and get up to speed quickly. Ask it how to set up a new project, how to run the tests, etc.

  > We tried to onboard ourselves with Potpie to the [**AgentOps**](https://github.com/AgentOps-AI/AgentOps) codebase and it worked like a charm: Video [here](https://youtu.be/_mPixNDn2r8).

- **Codebase Understanding** : Answer questions about any library you're integrating, explain functions, features, and architecture.

  > We used the Q&A agent to understand the underlying working of a feature of the [**CrewAI**](https://github.com/CrewAIInc/CrewAI) codebase that was not documented in official docs: Video [here](https://www.linkedin.com/posts/dhirenmathur_what-do-you-do-when-youre-stuck-and-even-activity-7256704603977613312-8X8G).

- **Low Level Design** : Get detailed implementation plans for new features or improvements before writing code.

  > We fed an open issue from the [**Portkey-AI/Gateway**](https://github.com/Portkey-AI/Gateway) project to this agent to generate a low level design for it: Video [here](https://www.linkedin.com/posts/dhirenmathur_potpie-ai-agents-vs-llms-i-am-extremely-activity-7255607456448286720-roOC).

- **Reviewing Code Changes** : Understand the functional impact of changes and compute the blast radius of modifications.

- **Debugging** : Get step-by-step debugging guidance based on stacktraces and codebase context.

- **Testing** : Generate contextually aware unit and integration test plans and test code that understand your codebase's structure and purpose.

## Extensions & Integrations

### VSCode Extension

Bring the power of Potpie's AI agents directly into your development environment.

- **Direct Integration** : Access all Potpie agents without leaving your editor
- **Quick Setup** : Install directly from the [VSCode Marketplace](https://marketplace.visualstudio.com/items?itemName=PotpieAI.potpie-vscode-extension)
- **Seamless Workflow** : Ask questions, get explanations, and implement suggestions right where you code

### Slack Integration

Bring your custom AI agents directly into your team's communication hub.

- **Team Collaboration** : Access all Potpie agents where your team already communicates
- **Effortless Setup** : Install and configure in under 2 minutes. [Docs](https://docs.potpie.ai/extensions/slack)
- **Contextual Assistance** : Get answers, code solutions, and project insights directly in Slack

👉 Install the Potpie Slack App: [Here](https://slack.potpie.ai/slack/install)

### API Access

Access Potpie Agents through an API key for CI/CD workflows and automation. See the [API documentation](https://docs.potpie.ai/agents/api-access).

### Tool Integration

Edit or add tools in `app/modules/intelligence/tools` and initialize them in `app/modules/intelligence/tools/tool_service.py`.

## Community & Support

- [GitHub Issues](https://github.com/potpie-ai/potpie/issues). Best for: bugs and errors you encounter using Potpie.
- [Discord](https://discord.gg/ryk5CMD5v6). Best for: sharing your projects and hanging out with the community.
- [Email Support](https://potpie.ai). Best for: problems with your setup or infrastructure.


See [Contributing Guide](./contributing.md) for more details.

## License

This project is licensed under the Apache 2.0 License — see the [LICENSE](LICENSE) file for details.

## Contributors

Thanks for spending your time helping build Potpie. Keep rocking 🥂

<img src="https://contributors-img.web.app/image?repo=potpie-ai/potpie" alt="Contributors"/>
