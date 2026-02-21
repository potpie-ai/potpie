
<p align="center">
  <a href="https://potpie.ai?utm_source=github">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="./assets/readme_logo_dark.svg" />
      <source media="(prefers-color-scheme: light)" srcset="./assets/readme_logo_light.svg" />
      <img src="./assets/logo_light.svg"  alt="Potpie AI logo" />
    </picture>
  </a>
</p>


# Potpie

[Potpie](https://potpie.ai) turns your entire codebase into a **knowledge graph** - a structural index of every file, class, and function, capturing all their relationships and what each part of the code does in context of everything else. AI agents built on this graph can reason about your code with the precision of someone who wrote it - from debugging to feature development.


<p align="center">
<img width="700" alt="Potpie Dashboard" src="./assets/dashboard.gif" />

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

#### Now set up Potpie Frontend

```bash
cd potpie-ui

cp .env.template .env

pnpm build && pnpm start
```


---

## How it works?

Potpie parses your repository into a **knowledge graph** stored in Neo4j - capturing every file, function, class, and the relationships between them. Agents read directly from this graph to answer questions and complete tasks grounded in your actual code.

**Architecture**

<p align="center">
  <img src="./assets/architecture.svg" alt="Potpie Architecture" width="900"/>
</p>

- [**FastAPI**](https://fastapi.tiangolo.com/) serves as the API layer — all requests enter through `localhost:8001` with CORS, Logfire tracing, and optional Sentry error tracking.
- [**Firebase Auth**](https://firebase.google.com/docs/auth) handles production authentication. In development mode a dummy user is created locally — no Firebase needed.
- [**Celery Worker**](https://docs.celeryq.dev/) with [Redis](https://redis.io/) as the broker handles async repo parsing — cloning, AST extraction, and knowledge graph construction run entirely in the background.
- [**Conversation Service**](https://docs.potpie.ai/conversations) manages chat sessions and agent memory across multi-turn interactions.
- [**Agent Router**](https://docs.potpie.ai/agents/introduction) dispatches prompts to the correct pre-built or custom agent based on intent.
- [**Tool Service**](https://docs.potpie.ai/tools) exposes callable functions to agents — code search, file fetch, knowledge graph queries, web tools, and more.
- [**Neo4j Knowledge Graph**](https://neo4j.com/) stores your codebase as a property graph — functions, classes, files, imports, and call relationships — the backbone of every agent's context.
- [**PostgreSQL**](https://www.postgresql.org/) stores users, projects, conversations, and message history.


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

      
      uv sync
    

This will create a `.venv` directory and install all dependencies from `pyproject.toml`

#### GitHub Authentication Setup

Potpie supports multiple authentication methods for accessing GitHub repositories:

##### For GitHub.com Repositories:

**Option 1: GitHub App (Recommended for Production)**
  - Create a GitHub App in your organization
  - Set environment variables:
    ```bash
    GITHUB_APP_ID=your-app-id
    GITHUB_PRIVATE_KEY=your-private-key
    ```

**Option 2: Personal Access Token (PAT) Pool**
  - Create one or more GitHub PATs with `repo` scope
  - Set environment variable (comma-separated for multiple tokens):
    ```bash
    GH_TOKEN_LIST=ghp_token1,ghp_token2,ghp_token3
    ```
  - Potpie will randomly select from the pool for load balancing
  - **Rate Limit**: 5,000 requests/hour per token (authenticated)

**Option 3: Unauthenticated Access (Public Repos Only)**
  - No configuration needed
  - Automatically used as fallback for public repositories
  - **Rate Limit**: 60 requests/hour per IP (very limited)

##### For Self-Hosted Git Servers (GitBucket, GitLab, etc.):

      Set the following environment variables:
      
    
      CODE_PROVIDER=github   # Options: github, gitbucket
      CODE_PROVIDER_BASE_URL=http://your-git-server.com/api/v3
      CODE_PROVIDER_TOKEN=your-token
    

**Important**: `GH_TOKEN_LIST` tokens are always used for GitHub.com, regardless of `CODE_PROVIDER_BASE_URL`.

2. **Start Potpie**

   To start all Potpie services:

   ```bash
   chmod +x scripts/start.sh
   ./scripts/start.sh
   ```

   This will:
   - Start required Docker services
   - Wait for PostgreSQL to be ready
   - Apply database migrations
   - Start the FastAPI application
   - Start the Celery worker

    **Optional: Logfire Tracing Setup**

      To monitor LLM traces and agent operations with Pydantic Logfire:

      1. Get a Logfire token from https://logfire.pydantic.dev
      2. Add it to your `.env` file:
      ```bash
      LOGFIRE_TOKEN=your_token_here
      ```
      3. Tracing is automatically initialized when Potpie starts. View traces at https://logfire.pydantic.dev

      **Note:** Set `LOGFIRE_SEND_TO_CLOUD=false` in your `.env` to disable sending traces to Logfire cloud.

3. **Stop Potpie**

   To stop all Potpie services:

   ```bash
   ./scripts/stop.sh
   ```

   **Windows**

   ```powershell
   ./stop.ps1
   ```

   This will gracefully stop:
   - The FastAPI application
   - The Celery worker
   - All Docker Compose services

## 🤖 Potpie's Prebuilt Agents

Potpie offers a suite of specialized codebase agents for automating and optimizing key aspects of software development:

<table>
  <tr>
    <td valign="top" width="50%">
      <h3>Debugging Agent</h3>
      <p>Automatically analyzes stacktraces and provides step-by-step debugging guidance specific to your codebase - not generic advice.</p>
      <a href="https://docs.potpie.ai/agents/debugging-agent"><img src="https://img.shields.io/badge/Learn%20More-Docs-22c55e?style=flat-square" alt="Docs"/></a>
    </td>
    <td valign="top" width="50%">
      <h3>Codebase Q&amp;A Agent</h3>
      <p>Answers questions about your codebase and explains functions, features, and architecture from first principles.</p>
      <a href="https://docs.potpie.ai/agents/codebase-qa-agent"><img src="https://img.shields.io/badge/Learn%20More-Docs-22c55e?style=flat-square" alt="Docs"/></a>
    </td>
  </tr>
  <tr>
    <td valign="top">
      <h3>Code Changes Agent</h3>
      <p>Analyzes code changes, identifies affected APIs, and suggests improvements before you merge — catching regressions early.</p>
      <a href="https://docs.potpie.ai/agents/code-changes-agent"><img src="https://img.shields.io/badge/Learn%20More-Docs-22c55e?style=flat-square" alt="Docs"/></a>
    </td>
    <td valign="top">
      <h3>Code Generation Agent</h3>
      <p>Generates code for new features, refactors existing code, and suggests optimizations grounded in your actual codebase.</p>
      <a href="https://docs.potpie.ai/agents/code-generation-agent"><img src="https://img.shields.io/badge/Learn%20More-Docs-22c55e?style=flat-square" alt="Docs"/></a>
    </td>
  </tr>
</table>


### Custom Agents

With Custom Agents, you can design personalized tools that handle repeatable tasks with precision. Define:

- **System Instructions** - The agent's task, goal, and expected output
- **Tasks** - Individual steps for job completion
- **Tools** - Functions for querying the knowledge graph or retrieving code

```bash
curl -X POST "http://localhost:8001/api/v1/custom-agents/agents/auto" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "An agent that takes stacktrace as input and gives root cause analysis and proposed solution as output"}'
```

Read more in our [documentation](https://docs.potpie.ai/open-source/agents/create-agent-from-prompt).



## Use Cases

<table>
  <tr>
    <td valign="top" width="50%">
      <h3>Onboarding</h3>
      <p>Get new developers productive in hours, not weeks. Potpie maps your architecture, entry points, and setup flows so anyone can hit the ground running.</p>
    </td>
    <td valign="top" width="50%">
      <h3>Codebase Q&amp;A</h3>
      <p>Ask anything about your codebase — functions, data flows, design decisions. Get precise answers grounded in your actual code, not guesses.</p>
    </td>
  </tr>
  <tr>
    <td valign="top">
      <h3>Debugging</h3>
      <p>Paste a stacktrace. Get a root-cause analysis and step-by-step fix path pinpointed to your code — not generic troubleshooting advice.</p>
    </td>
    <td valign="top">
      <h3>Code Review</h3>
      <p>Before you merge, understand the blast radius of your changes. Potpie surfaces affected APIs, downstream impacts, and potential regressions.</p>
    </td>
  </tr>
  <tr>
    <td valign="top">
      <h3>Test Generation</h3>
      <p>Generate unit and integration tests that understand your code structure — not boilerplate. Covers edge cases your manual tests would miss.</p>
    </td>
    <td valign="top">
      <h3>Feature Planning</h3>
      <p>Turn a requirement or open issue into a low-level implementation plan — with component breakdown, API surface, and suggested code structure.</p>
    </td>
  </tr>
</table>

<p align="center">
  <a href="https://docs.potpie.ai/usecases">
    <img src="https://img.shields.io/badge/Explore%20All%20Use%20Cases-Docs-22c55e?style=for-the-badge" alt="Explore All Use Cases"/>
  </a>
</p>

## Extensions &amp; Integrations

<table>
  <tr>
    <td valign="top">
      <h3>VSCode Extension</h3>
      <p>Use Potpie's AI agents directly inside your editor — no tab switching, no copy-pasting. Ask questions, get explanations, and ship code without leaving VSCode.</p>
      <ul>
        <li>Agents understand your open repo instantly</li>
        <li>Works with any codebase — local or remote</li>
      </ul>
      <a href="https://marketplace.visualstudio.com/items?itemName=PotpieAI.potpie-vscode-extension">
        <img src="https://img.shields.io/badge/Install-VSCode%20Extension-0078d7?style=for-the-badge&logo=visualstudiocode&logoColor=white" alt="Install VSCode Extension"/>
      </a>
    </td>
  </tr>
  <tr><td><hr/></td></tr>
  <tr>
    <td valign="top">
      <h3>Slack Integration</h3>
      <p>Bring Potpie into your team's Slack workspace. Debug code, answer codebase questions, and get project insights — all in threads your team already uses.</p>
      <ul>
        <li>Set up in under 2 minutes</li>
        <li>Works with your custom configured agents</li>
      </ul>
      <a href="https://slack.potpie.ai/slack/install">
        <img src="https://img.shields.io/badge/Add%20to-Slack-4A154B?style=for-the-badge&logo=slack&logoColor=white" alt="Add to Slack"/>
      </a>
      &nbsp;
      <a href="https://docs.potpie.ai/extensions/slack">
        <img src="https://img.shields.io/badge/Setup-Guide-22c55e?style=for-the-badge" alt="Setup Guide"/>
      </a>
    </td>
  </tr>
  <tr><td><hr/></td></tr>
  <tr>
    <td valign="top">
      <h3>API Access</h3>
      <p>Integrate Potpie into CI/CD pipelines and automation workflows via API key. Trigger agents programmatically to fit your existing DevOps setup.</p>
      <a href="https://docs.potpie.ai/agents/api-access">
        <img src="https://img.shields.io/badge/View-API%20Docs-22c55e?style=for-the-badge" alt="API Docs"/>
      </a>
    </td>
  </tr>
  <tr><td><hr/></td></tr>
  <tr>
    <td valign="top">
      <h3>Custom Tool Integration</h3>
      <p>Extend Potpie's capabilities by building and registering your own tools. Add files to <code>app/modules/intelligence/tools/</code> and register them in <code>tool_service.py</code>.</p>
    </td>
  </tr>
</table>


## Community & Support

- [GitHub Issues](https://github.com/potpie-ai/potpie/issues). Best for: bugs and errors you encounter using Potpie.
- [Discord](https://discord.gg/ryk5CMD5v6). Best for: sharing your projects and hanging out with the community.
- [Email Support](https://potpie.ai). Best for: problems with your setup or infrastructure.


See the [Contributing Guide](https://github.com/potpie-ai/potpie/blob/main/.github/CONTRIBUTING.md) for more details.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## Contributors

Thanks for spending your time helping build Potpie. Keep rocking 🥂

Want to contribute? Read the [Contributing Guide](https://github.com/potpie-ai/potpie/blob/main/.github/CONTRIBUTING.md) to get started.

<a href="https://github.com/potpie-ai/potpie/graphs/contributors">
  <img src="https://contributors-img.web.app/image?repo=potpie-ai/potpie" alt="Contributors"/>
</a>
