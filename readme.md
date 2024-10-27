# Potpie: Build AI agents for your codebase in minutes.

Potpie deeply understands your codebase by breaking down your code into its constituting parts and building a knowledge graph out of your code’s components. We generate inferences at every level of your codebase so that we can comprehensively answer questions about your codebase.

Potpie also provides purpose built agents that are expert on your codebase to perform engineering tasks for you, and also provides the platform for you to build your own custom agents using tools that interface with the knowledge graph.

---
## Table of Contents

- [What Makes Potpie Agents Unique?](#what-makes-potpie-agents-unique)
- [Potpie's Tooling System](#potpie's-tooling-system)
- [The Power of Custom Agents](#the-power-of-custom-agents-coming-soon)
- [Contributing](#contributing)
- [License](#license)


## What Makes Potpie Agents Unique?

Potpie offers a suite of specialized agents that empower developers by automating and optimizing key aspects of software development:

- **Debugging Agent**: Automatically analyzes stacktraces and provides debugging steps specific to your codebase.  
- **Codebase Q&A Agent**: Answers questions about your codebase and explains functions, features, and architecture.  
- **Code Changes Agent**: Analyzes code changes, identifies affected APIs, and suggests improvements before merging.  
- **Integration Test Agent**: Generates integration test plans and code for flows to ensure components work together properly.  
- **Unit Test Agent**: Automatically creates unit test plan and code for individual functions to enhance test coverage.  
- **LLD Agent**: Creates a low level design for implementing a new feature by providing functional requirements to this agent.

Potpie's agents work by leveraging tools that interact with the knowledge graph. The knowledge graph is a meticulously constructed graph of the codebase tracking relationships between functions, files, classes of the codebase, stored in neo4j. We generate inferences for each node and embed and store it in the graph to perform similarity search for the user query. Other tools look up project structure, fetch code from github, fetch code flow from graph etc 

---

## Potpie's Tooling System

Potpie provides a set of tools that agents can use to interact with the knowledge graph and the underlying infrastructure. These tools are vital for creating custom agents and for performing highly contextual tasks with precision.

#### Available Tools:
- **get_code_from_probable_node_name**: Retrieves code snippets based on a probable node name.
- **get_code_from_node_id**: Fetches code associated with a specific node ID.
- **get_code_from_multiple_node_ids**: Retrieves code snippets for multiple node IDs simultaneously.
- **ask_knowledge_graph_queries**: Executes vector similarity searches to obtain relevant information from the knowledge graph.
- **get_nodes_from_tags**: Retrieves nodes tagged with specific keywords from the knowledge graph.
- **get_code_graph_from_node_id/name**: Fetches code graph structures for a specific node ID or name.
- **change_detection**: Detects changes in the current branch compared to the default branch.


These tools are the foundation for the custom agents you create, allowing them to intelligently access and manipulate your codebase efficiently.

---
## The Power of Custom Agents [Coming Soon]

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

## Contributing

We welcome contributions from the community. Contributions can be of the form: 
1. Documentation : Help improve our docs! If you fixed a problem, chances are others faced it too.
2. Code : Help us make improvements to existing features and build new features for Potpie. 
3. Tests :  Help us make Potpie resilient by contributing tests.

To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a Pull Request.

Refer to the [Contributing Guide](./contributing.md) for more details.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## 💪 Thanks To All Contributors

Thanks a lot for spending your time helping build Potpie. Keep rocking 🥂

<img src="https://contributors-img.web.app/image?repo=potpie-ai/potpie" alt="Contributors"/>
