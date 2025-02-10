<p align="center">
  <a href="https://potpie.ai?utm_source=github">
    <img src="https://github.com/user-attachments/assets/1a0b9824-833b-4c0a-b56d-ede5623295ca" width="318px" alt="Momentum logo" />
  </a>
</p>

<br/>
<p align="center">
<a href="https://trendshift.io/repositories/12918" target="_blank"><img src="https://trendshift.io/api/badge/repositories/12918" alt="potpie-ai%2Fpotpie | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</br>
  <br />
  <a href="https://app.potpie.ai" rel="dofollow">App</a> | <a href="https://docs.potpie.ai" rel="dofollow">Documentation</a> | <a href="https://docs.potpie.ai/open-source"  rel="dofollow">API Reference</a> | <a href="https://app.potpie.ai/newchat?repo=potpie-ai/potpie&branch=main" rel="dofollow">Chat with 🥧 Repo</a>
  <br />

  </p>

<p align="center">

  <a href="https://github.com/potpie-ai/potpie/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/potpie-ai/potpie" alt="Apache 2.0">
  </a>

  <a href="https://github.com/potpie-ai/potpie">
    <img src="https://img.shields.io/github/stars/potpie-ai/potpie" alt="GitHub Repo stars">
  </a>
  
</br>


<a href="https://discord.gg/ryk5CMD5v6">
    <img src="https://img.shields.io/badge/Join%20our-Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Join our Discord">
</a>
</br>
<a href="https://twitter.com/intent/tweet?text=I%20created%20custom%20engineering%20agents%20for%20my%20codebase%20in%20minutes%20with%20potpie.ai%20@potpiedotai%20!🥧">
    <img alt="tweet" src="https://img.shields.io/twitter/url/http/shields.io.svg?style=social">
</a>

</p>

<h1 align="center">

Prompt-To-Agent: Create custom engineering agents for your code
</h1>

Potpie is an open-source platform that creates AI agents specialized in your codebase, enabling automated code analysis, testing, and development tasks. By building a comprehensive knowledge graph of your code, Potpie's agents can understand complex relationships and assist with everything from debugging to feature development.

<p align="center">
<img width="1506" alt="Screenshot 2025-01-09 at 2 18 18 PM" src="https://github.com/user-attachments/assets/a400b48f-dc4c-47b1-a42b-26eaf062adb2" />

</p>

## 📚 Table of Contents
- [🥧 Why Potpie?](#why-potpie)
- [🤖 Our Prebuilt Agents](#prebuilt-agents)
- [🛠️ Tooling](#potpies-tooling-system)
- [🚀 Getting Started](#getting-started)
- [💡 Use Cases](#use-cases)
- [🛠️ Custom Agents](#custom-agents-upgrade)
- [🗝️ Accessing Agents via API Key](#accessing-agents-via-api-key)
- [🎨 Make Potpie Your Own](#make-potpie-your-own)
- [🤝 Contributing](#contributing)
- [📜 License](#license)
- [💪 Contributors](#-thanks-to-all-contributors)


## 🥧 Why Potpie?
- 🧠 **Deep Code Understanding**: Built-in knowledge graph captures relationships between code components
- 🤖 **Pre-built & Custom Agents**: Ready-to-use agents for common tasks + build your own
- 🔄 **Seamless Integration**: Works with your existing development workflow
- 📈 **Flexible**: Handles codebases of any size or language



## 🤖 Potpie's Prebuilt Agents

Potpie offers a suite of specialized codebase agents for automating and optimizing key aspects of software development:

- **Debugging Agent**: Automatically analyzes stacktraces and provides debugging steps specific to your codebase.
- **Codebase Q&A Agent**: Answers questions about your codebase and explains functions, features, and architecture.
- **Code Changes Agent**: Analyzes code changes, identifies affected APIs, and suggests improvements before merging.
- **Integration Test Agent**: Generates integration test plans and code for flows to ensure components work together properly.
- **Unit Test Agent**: Automatically creates unit test plan and code for individual functions to enhance test coverage.
- **LLD Agent**: Creates a low level design for implementing a new feature by providing functional requirements to this agent.
- **Code Generation Agent**: Generates code for new features, refactors existing code, and suggests optimizations.

## 🛠️ Potpie's Tooling System

Potpie provides a set of tools that agents can use to interact with the knowledge graph and the underlying infrastructure:

- **get_code_from_probable_node_name**: Retrieves code snippets based on a probable node name.
- **get_code_from_node_id**: Fetches code associated with a specific node ID.
- **get_code_from_multiple_node_ids**: Retrieves code snippets for multiple node IDs simultaneously.
- **ask_knowledge_graph_queries**: Executes vector similarity searches to obtain relevant information.
- **get_nodes_from_tags**: Retrieves nodes tagged with specific keywords.
- **get_code_graph_from_node_id/name**: Fetches code graph structures for a specific node.
- **change_detection**: Detects changes in the current branch compared to the default branch.
- **get_code_file_structure**: Retrieves the file structure of the codebase.

## 🚀 Getting Started

### Prerequisites
- Docker installed and running
- OpenAI API key
- Git installed (for repository access)
- Python 3.10.x

### Setup Steps

**Install Python 3.10**
   - Download and install Python 3.10 from the official Python website:
     https://www.python.org/downloads/release/python-3100/

1. **Prepare Your Environment**
   - Create a `.env` file based on the `.env.template`
   - Add the following required configurations:
      ```bash
      isDevelopmentMode=enabled
      ENV=development
      OPENAI_API_KEY=<your-openai-key>
      POSTGRES_SERVER=postgresql://postgres:mysecretpassword@localhost:5432/momentum
      NEO4J_URI=bolt://127.0.0.1:7687
      NEO4J_USERNAME=neo4j
      NEO4J_PASSWORD=mysecretpassword
      REDISHOST=127.0.0.1
      REDISPORT=6379
      BROKER_URL=redis://127.0.0.1:6379/0
      CELERY_QUEUE_NAME=dev
      defaultUsername=defaultuser
      PROJECT_PATH=projects #repositories will be downloaded/cloned to this path on your system.
      ```
   -  Create a Virtual Environment using Python 3.10:
      ```bash
      python3.10 -m venv venv
      source venv/bin/activate
      ```
      alternatively, you can also use the `virtualenv` library.

    - Install dependencies in your venv:
      ```bash
      pip install -r requirements.txt
      ```

2. **Start Potpie**
   ```bash
   chmod +x start.sh
   ./start.sh
   ```

3. **Authentication Setup** (Skip this step in development mode)
   ```bash
   curl -X POST 'http://localhost:8001/api/v1/login' \
     -H 'Content-Type: application/json' \
     -d '{
       "email": "your-email",
       "password": "your-password"
     }'
   # Save the bearer token from the response for subsequent requests
   ```

4. **Initialize Repository Parsing**
   ```bash
   # For development mode:
   curl -X POST 'http://localhost:8001/api/v1/parse' \
     -H 'Content-Type: application/json' \
     -d '{
       "repo_path": "path/to/local/repo",
       "branch_name": "main"
     }'

   # For production mode:
   curl -X POST 'http://localhost:8001/api/v1/parse' \
     -H 'Content-Type: application/json' \
     -d '{
       "repo_name": "owner/repo-name",
       "branch_name": "main"
     }'
   # Save the project_id from the response
   ```

5. **Monitor Parsing Status**
   ```bash
   curl -X GET 'http://localhost:8001/api/v1/parsing-status/your-project-id'
   # Wait until parsing is complete
   ```

6. **View Available Agents**
   ```bash
   curl -X GET 'http://localhost:8001/api/v1/list-available-agents/?list_system_agents=true'
   # Note down the agent_id you want to use
   ```

7. **Create a Conversation**
   ```bash
   curl -X POST 'http://localhost:8001/api/v1/conversations/' \
     -H 'Content-Type: application/json' \
     -d '{
       "user_id": "your_user_id",
       "title": "My First Conversation",
       "status": "active",
       "project_ids": ["your-project-id"],
       "agent_ids": ["chosen-agent-id"]
     }'
   # Save the conversation_id from the response
   ```

8. **Start Interacting with Your Agent**
   ```bash
   curl -X POST 'http://localhost:8001/api/v1/conversations/your-conversation-id/message/' \
     -H 'Content-Type: application/json' \
     -d '{
       "content": "Your question or request here"
     }'
   ```

9. **View Conversation History** (Optional)
   ```bash
   curl -X GET 'http://localhost:8001/api/v1/conversations/your-conversation-id/messages/?start=0&limit=10'
   ```

## 💡 Use Cases

- **Onboarding**: For developers new to a codebase, the codebase QnA agent helps them understand the codebase and get up to speed quickly. Ask it how to setup a new project, how to run the tests etc
>We tried to onboard ourselves with Potpie to the [**AgentOps**](https://github.com/AgentOps-AI/AgentOps) codebase and it worked like a charm : Video [here](https://youtu.be/_mPixNDn2r8).

- **Codebase Understanding**: Answer questions about any library you're integrating, explain functions, features, and architecture.
>We used the Q&A agent to understand the underlying working of a feature of the [**CrewAI**](https://github.com/CrewAIInc/CrewAI) codebase that was not documented in official docs : Video [here](https://www.linkedin.com/posts/dhirenmathur_what-do-you-do-when-youre-stuck-and-even-activity-7256704603977613312-8X8G).

- **Low Level Design**: Get detailed implementation plans for new features or improvements before writing code.
>We fed an open issue from the [**Portkey-AI/Gateway**](https://github.com/Portkey-AI/Gateway) project to this agent to generate a low level design for it: Video [here](https://www.linkedin.com/posts/dhirenmathur_potpie-ai-agents-vs-llms-i-am-extremely-activity-7255607456448286720-roOC).

- **Reviewing Code Changes**: Understand the functional impact of changes and compute the blast radius of modifications.
>Here we analyse a PR from the [**mem0ai/mem0**](https://github.com/mem0ai/mem0) codebase and understand its blast radius : Video [here](https://www.linkedin.com/posts/dhirenmathur_prod-is-down-three-words-every-activity-7257007131613122560-o4A7).

- **Debugging**: Get step-by-step debugging guidance based on stacktraces and codebase context.

- **Testing**: Generate contextually aware unit and integration test plans and test code that understand your codebase's structure and purpose.

## 🛠️ Custom Agents [Upgrade ✨](https://potpie.ai/pricing)

With Custom Agents, you can design personalized tools that handle repeatable tasks with precision. Key components include:
- **System Instructions**: Define the agent's task, goal, and expected output
- **Agent Information**: Metadata about the agent's role and context
- **Tasks**: Individual steps for job completion
- **Tools**: Functions for querying the knowledge graph or retrieving code

## 🗝️ Accessing Agents via API Key

You can access Potpie Agents through an API key, enabling integration into CI/CD workflows and other automated processes. For detailed instructions, please refer to the [Potpie API documentation](https://docs.potpie.ai/agents/api-access).

- **Generate an API Key**: Easily create an API key for secure access.
- **Parse Repositories**: Use the Parse API to analyze code repositories and obtain a project ID.
- **Monitor Parsing Status**: Check the status of your parsing requests.
- **Create Conversations**: Initiate conversations with specific agents using project and agent IDs adn get a conversation id.
- **Send Messages**: Communicate with agents by sending messages within a conversation.

## 🎨 Make Potpie Your Own

Potpie is designed to be flexible and customizable. Here are key areas to personalize your own deployment:

### 1. System Prompts Configuration
Modify prompts in `app/modules/intelligence/prompts/system_prompt_setup.py`

### 2. Add New Agents
Create new agents in `app/modules/intelligence/agents/chat_agents` and `app/modules/intelligence/agents/agentic_tools`

### 3. Agent Behavior Customization
Modify guidelines within each agent's prompt in the `app/modules/intelligence/agents` directory

### 4. Tool Integration
Edit or add tools in the `app/modules/intelligence/tools` directory

## 🤝 Contributing

We welcome contributions! To contribute:
1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Make your changes
4. Commit (`git commit -m 'Add new feature'`)
5. Push to the branch (`git push origin feature-branch`)
6. Open a Pull Request

See [Contributing Guide](./contributing.md) for more details.

## 📜 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## 💪 Thanks To All Contributors

Thanks for spending your time helping build Potpie. Keep rocking 🥂

<img src="https://contributors-img.web.app/image?repo=potpie-ai/potpie" alt="Contributors"/>
