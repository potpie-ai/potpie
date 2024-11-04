<p align="center">
  <a href="https://potpie.ai?utm_source=github">
    <img src="https://github.com/user-attachments/assets/1a0b9824-833b-4c0a-b56d-ede5623295ca" width="318px" alt="Momentum logo" />
  </a>
</p>

<br/>

<p align="center">
  <a href="https://github.com/potpie-ai/potpie/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/potpie-ai/potpie" alt="Apache 2.0">
  </a>
</p>

<h1 align="center">
AI agents for your codebase in minutes.
</h1>

<p align="center">
  <br />
  <a href="https://app.potpie.ai" rel="dofollow"><strong>Get started!</strong></a>
  <br />
  <a href="https://docs.potpie.ai" rel="dofollow">Documentation</a>
  <br />


Potpie parses and understands your codebase by building a knowledge graph out of your code’s components.
It provides pre-built agents that are expert on your codebase to perform common engineering tasks for you, and also provides the platform for you to build your own custom agents.

<p align="center">
  <a href="https://app.potpie.ai">
    <img src="https://github.com/user-attachments/assets/a808e22d-af9b-417f-a904-6410b01852e0" alt="Potpie Dashboard" />
  </a>
</p>

---
## Table of Contents

- [What are Codebase Agents?](#why-agents)
- [Our Prebuilt Agents](#prebuilt-agents)
- [Tooling](#potpies-tooling-system)
- [Getting Started](#getting-started)
- [Use Cases](#use-cases)
- [Custom Agents](#custom-agents-upgrade)
- [Make Potpie Your Own](#make-potpie-your-own)
- [Contributing](#contributing)
- [License](#license)
- [Contributors](#-thanks-to-all-contributors)


## What are Codebase Agents?

AI agents are autonomous tools that have the ability to reason, take decisions and perform actions on their own. They are provided with 'tools' that they can use to perform tasks. Agents are iterative in nature and build on top of the results of the previous iteration in order to perform any task assigned to them.

Software development is a similarly iterative process and agents can be used to automate and optimize key aspects of software development.
Things that developers do daily, like dbugging, can be broken down into a series of iterative steps that can be automated by agents.
For example, debugging can be broken down into:
1. Understanding the stacktrace
2. Understanding the code around the stacktrace
3. Coming up with a hypothesis
4. Testing the hypothesis
5. Repeating the above steps until the bug is fixed

In order to perform these steps, an agent would need to understand the codebase, the code around the stacktrace, the flow of the code, the project structure etc.

Potpie parses your codebase and builds a graph tracking relationships between functions, files, classes, etc. We generate inferences for each node and embed and store it in the graph. This can be used to curate the correct context by performing a similarity search based on users query. The graph can also be queried to understand the code flow, it can be queried to understand the project structure etc.

This allows Potpie's agents to understand the codebase and reason about the code.

## Potpie's Prebuilt Agents

Potpie offers a suite of specialized codebase agents for automating and optimizing key aspects of software development:

- **Debugging Agent**: Automatically analyzes stacktraces and provides debugging steps specific to your codebase.
- **Codebase Q&A Agent**: Answers questions about your codebase and explains functions, features, and architecture.
- **Code Changes Agent**: Analyzes code changes, identifies affected APIs, and suggests improvements before merging.
- **Integration Test Agent**: Generates integration test plans and code for flows to ensure components work together properly.
- **Unit Test Agent**: Automatically creates unit test plan and code for individual functions to enhance test coverage.
- **LLD Agent**: Creates a low level design for implementing a new feature by providing functional requirements to this agent.

Potpie's agents leverage tools that interact with your codebase's knowledge graph stored in neo4j. These tools look up project structure, fetch code from github, fetch code flow from graph etc

### Potpie's Tooling System

Potpie provides a set of tools that agents can use to interact with the knowledge graph and the underlying infrastructure. These tools are vital for creating custom agents and for performing highly contextual tasks with precision.

#### Available Tools:
- **get_code_from_probable_node_name**: Retrieves code snippets based on a probable node name.
- **get_code_from_node_id**: Fetches code associated with a specific node ID.
- **get_code_from_multiple_node_ids**: Retrieves code snippets for multiple node IDs simultaneously.
- **ask_knowledge_graph_queries**: Executes vector similarity searches to obtain relevant information from the knowledge graph.
- **get_nodes_from_tags**: Retrieves nodes tagged with specific keywords from the knowledge graph.
- **get_code_graph_from_node_id/name**: Fetches code graph structures for a specific node ID or name.
- **change_detection**: Detects changes in the current branch compared to the default branch.
- **get_code_file_structure**: Retrieves the file structure of the codebase.


These tools are the foundation for the custom agents you create, allowing them to intelligently access and manipulate your codebase efficiently.

---
## Getting Started

Refer to the [Getting Started Guide](./GETTING_STARTED.md) for detailed instructions on setting up Potpie and making your first agent work for you!

Once you have set up Potpie, you can get started with the following steps:

## Step 1: Logging in to get a bearer token
```bash
curl -X 'POST' \
  'http://localhost:8001/api/v1/login' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "email": "string",
  "password": "string"
}'
```
## Step 2: Submit a Parsing Request
Replace the repo name and branch name with the repo you want to talk to.
```bash
curl -X 'POST' \
  'http://localhost:8001/api/v1/parse' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "repo_name": "owner/repo-name",
  "branch_name": "branch-name"
}'
```
## Step 3: Check Parsing Status
Use the project id generated from previous request.
```bash
curl -X 'GET' \
  'http://localhost:8001/api/v1/parsing-status/project-id' \
  -H 'accept: application/json'
```
## Step 4: List Available Agents
```bash
curl -X 'GET' \
  'http://localhost:8001/api/v1/list-available-agents/?list_system_agents=true' \
  -H 'accept: application/json'
```
## Step 5: Create a Conversation
```bash
curl -X 'POST' \
  'http://localhost:8001/api/v1/conversations/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "user_id": "your_user_id",
  "title": "Conversation Title",
  "status": "active",
  "project_ids": [
    "project_id"
  ],
  "agent_ids": [
    "agent_id"
  ]
}'
```
## Step 6: Send Messages in a Conversation

This API returns a stream response for the
```bash
curl -X 'POST' \
  'http://localhost:8001/api/v1/conversations/1234/message/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "content": "Your message content here",
  "node_ids": [
    {
      "node_id": "node_identifier",
      "name": "node_name"
    }
  ]
}'
```
## Step 7: Get all the messages of a conversation
```bash
curl -X 'GET' \
  'http://localhost:8001/api/v1/conversations/conversation-id/
  messages/?start=0&limit=10' \
  -H 'accept: application/json'
```

---

## Use Cases

- **Onboarding**: For developers new to a codebase, the codebase QnA agent can help them understand the codebase and get up to speed quickly. Ask it how to setup a new project, how to run the tests etc
We tried to onboard ourselves with Potpie to the [**AgentOps**](https://github.com/AgentOps-AI/AgentOps) codebase and it worked like a charm : Video [here](https://youtu.be/_mPixNDn2r8).

- **Codebase Understanding**: Answer questions about any library you're trying to integrate, explain functions, features, and architecture. Ask it how a feature works, what does a function do, how to change a function etc.
We used the Q&A agent to understand the underlying working of a feature of the [**CrewAI**](https://github.com/CrewAIInc/CrewAI) codebase that was not documented in official docs : Video [here](https://www.linkedin.com/posts/dhirenmathur_what-do-you-do-when-youre-stuck-and-even-activity-7256704603977613312-8X8G).

- **Low level design**: Before writing that first line of code, it is important to know which files and functions need to be changed. This agent takes your functional requirements as an input and then generates a low level design for the feature. The output will consist of which files need to be changed, what all functions need to be added etc.
We fed an open issue from the [**Portkey-AI/Gateway**](https://github.com/Portkey-AI/Gateway) project to this agent to generate a low level design for it: Video [here](https://www.linkedin.com/posts/dhirenmathur_potpie-ai-agents-vs-llms-i-am-extremely-activity-7255607456448286720-roOC).

- **Reviewing code changes**: Every commit to the codebase has the potential to be a breaking change. Use this agent to understand the functional impact of the changes in the codebase. It compares the code changes with the default branch and computes the blast radius of the changes.
Here we analyse a PR from the [**mem0ai/mem0**](https://github.com/mem0ai/mem0) codebase and understand its blast radius : Video [here](https://www.linkedin.com/posts/dhirenmathur_prod-is-down-three-words-every-activity-7257007131613122560-o4A7).

- **Debugging**: Debugging is an iterative process that usually follows a set of well known steps. This agent emulates those steps and can be used to debug issues in the codebase. It takes a stacktrace as an input and then generates a list of steps that can be used to debug the issue.

- **Unit and Integration testing**: Use the Unit Test Agent to generate unit test plans and code for individual functions to enhance test coverage, similarly use the Integration Test Agent to generate integration test plans and code for flows to ensure components work together properly. These agents are highly contextual and will use the codebase graph to gather context for generating the tests.

---

## Custom Agents [Upgrade]

Potpie doesn’t stop at pre-built agents. With **Custom Agents**, developers can design personalized tools that handle repeatable tasks with precision. Whether it's generating boilerplate code, identifying security vulnerabilities, or suggesting optimizations, Potpie’s custom agents are flexible and built to adapt to your unique project requirements.

### Custom Agents for Advanced Workflows

Potpie’s cloud platform supports **Custom Agents**, enabling you to create agents that automate specific, repeatable tasks tailored to your project's unique requirements.

#### Key Components of Custom Agents
- **System Instructions**: Guidelines that define the agent's task, its goal, and the expected output.
- **Agent Information**: Metadata such as the agent’s role, goal, and task context.
- **Tasks**: The individual steps the agent will take to complete its job.
- **Tools**: Functions that allow the agent to perform its tasks, such as querying the knowledge graph or retrieving code snippets.


#### Example Use Cases:
- Automating code optimization and offering improvement suggestions.
- Identifying and reporting security vulnerabilities in the codebase.
- Automatically generating unit tests based on existing code logic.

---

## Make Potpie Your Own

Potpie is designed to be flexible and customizable. Here are key areas to personalize your own deployment:

### 1. System Prompts Configuration

Modify the system prompts to align with your organization's tone and terminology.

**Edit Prompt Text**: In `app/modules/intelligence/prompts/system_prompt_setup.py`, update the `system_prompts` lists to change the text for each agent.

### 2. Add New Agents
**Add New Agents**: Create new agents by referring existing agents in the `app/modules/intelligence/agents/chat_agents` and `app/modules/intelligence/agents/agentic_tools` directory.

### 3. Agent Behavior Customization

Adjust existing agent behaviors to suit your operational needs.

**Modify Guidelines**: Change the guidelines within each agent's prompt to emphasize specific aspects of your codebase. You can do this by editing the prompts in the crewai agents in the `app/modules/intelligence/agents` directory.


### 4. Tool Integration

Customize which tools are available to each agent based on your requirements.

**Edit existing tools**: Edit tools for your usecase by refactoring the existing tools in the `app/modules/intelligence/tools` directory.

**Add New Tools**: Add new tools by referring existing tools in the `app/modules/intelligence/tools` directory.


By customizing system prompts, agent behaviors, and tool integrations you can tailor Potpie to effectively meet your organization's unique needs and enhance your software development processes.


---

## Contributing

We welcome contributions from the community. Contributions can be of the form:
1. Documentation : Help improve our docs! If you fixed a problem, chances are others faced it too.
2. Code : Help us make improvements to existing features and build new features for Potpie.
3. Tests :  Help us make Potpie resilient by contributing tests.

To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Stage your changes (`git add <file>`), then commit them (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a Pull Request.

Refer to the [Contributing Guide](./contributing.md) for more details.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## 💪 Thanks To All Contributors

Thanks a lot for spending your time helping build Potpie. Keep rocking 🥂

<img src="https://contributors-img.web.app/image?repo=potpie-ai/potpie" alt="Contributors"/>
