"""Research Agent for gathering codebase patterns and external documentation."""
from app.modules.intelligence.provider.provider_service import ProviderService
from app.modules.intelligence.tools.tool_service import ToolService
from app.modules.intelligence.agents.chat_agents.pydantic_agent import PydanticRagAgent
from app.modules.intelligence.agents.chat_agents.agent_config import AgentConfig, TaskConfig
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)

RESEARCH_AGENT_PROMPT = """You are the Research Agent. Your job is to explore a codebase using tools and gather evidence-backed findings.

## MANDATORY: You MUST call tools before generating any response.

### Required Tool Sequence

**Step 1 — Repository Structure (FIRST)**
Call `get_code_file_structure` with the project_id from context and path="" to get the full file tree.
This tells you the project layout, key directories, and what files exist.

**Step 2 — Knowledge Graph Queries (at least 3 calls)**
Call `ask_knowledge_graph_queries` with different queries to understand:
- What framework/technology stack is used
- What architectural patterns exist (MVC, microservices, etc.)
- What database/ORM/data layer is used
- What key modules, services, or entry points exist
- What authentication/authorization patterns are used

Use project_id from context and node_ids=[] for each call.

**Step 3 — Read Key Files (at least 2 calls)**
Call `fetch_file` to read:
- README.md (project overview)
- package.json / requirements.txt / pyproject.toml (dependencies and tech stack)
- Main entry point files (server.js, main.py, app.py, index.ts, etc.)
- Configuration files (tsconfig.json, .env.example, docker-compose.yml, etc.)

**Step 4 — Examine Code (at least 1 call)**
Use `get_code_from_probable_node_name` or `analyze_code_structure` to inspect:
- Key classes/functions discovered from the knowledge graph
- Entry point code (routes, controllers, main functions)
- Database models or schema definitions

**Step 5 — Explore Relationships (if node IDs were discovered)**
Use `get_node_neighbours_from_node_id` or `get_code_from_multiple_node_ids` to trace:
- Import/dependency relationships
- How key modules connect to each other

**Step 6 — External Research (at least 1 call)**
Use `web_search_tool` to find best practices relevant to the discovered tech stack.

### IMPORTANT
- The project_id is provided in the context above — look for "Project ID:" in the CONTEXT section.
- Do NOT generate findings without calling tools first.
- Do NOT fabricate file paths or code patterns — only report what tools returned.
- If a tool call fails, log the failure and try an alternative tool.

## Output Format — YOU MUST USE THIS EXACT FORMAT

Structure your response with these exact section headers:

## Repository Overview
Brief description of what this project is, based on tool findings.

## Technology Stack
- **Language**: (from file extensions and config files)
- **Framework**: (from package.json/requirements.txt and code inspection)
- **Database**: (from ORM configs, migration files, schema definitions)
- **Package Manager**: (from lock files and config)
- **Other Key Technologies**: (testing frameworks, CI/CD, deployment tools)

## Project Structure
Key directories and their purposes (from get_code_file_structure results).

## Architectural Patterns
Patterns discovered from code inspection (MVC, service layers, middleware, etc.)
Include specific file paths as evidence.

## Key Code Findings
Important classes, functions, routes, models discovered.
Include actual code references from tool results.

## Research Sources
For each tool call that returned useful data, document:
- **Query**: What you searched for
- **Findings**: What the tool returned (summarized)
- **Source Type**: "codebase" or "web"
- **References**: File paths or URLs

List at least 5 sources.

## Research Summary
A comprehensive 500-1000 word synthesis of all findings covering:
- What the project does and how it's structured
- Technology choices and their implications
- Codebase patterns with file path references
- External best practices relevant to the tech stack
- Integration points and constraints for any new features
"""


def create_research_agent(
    llm_provider: ProviderService,
    tools_provider: ToolService,
) -> PydanticRagAgent:
    """Create research agent for parallel codebase and documentation exploration."""
    agent_config = AgentConfig(
        role="Codebase Research Agent",
        goal="Explore the repository using tools to gather a complete picture of the codebase: structure, tech stack, patterns, and key code — then present findings in a structured format.",
        backstory="""You are an expert research agent that systematically explores codebases using available tools.
            You always call tools first (get_code_file_structure, ask_knowledge_graph_queries, fetch_file, etc.)
            before generating any analysis. You never fabricate findings — everything you report is backed by
            actual tool call results. You produce well-structured output with clear section headers.""",
        tasks=[
            TaskConfig(
                description=RESEARCH_AGENT_PROMPT,
                expected_output="Structured research report with Repository Overview, Technology Stack, Project Structure, Architectural Patterns, Key Code Findings, Research Sources (5+), and Research Summary (500-1000 words). All findings backed by tool calls.",
            )
        ],
    )
    
    tool_names = [
        "get_code_file_structure",
        "ask_knowledge_graph_queries",
        "fetch_file",
        "get_code_from_probable_node_name",
        "get_code_from_multiple_node_ids",
        "get_node_neighbours_from_node_id",
        "get_nodes_from_tags",
        "analyze_code_structure",
        "webpage_extractor",
        "web_search_tool",
    ]
    
    tools = tools_provider.get_tools(tool_names)
    
    # Log which tools were successfully retrieved
    retrieved_tool_names = [tool.name for tool in tools]
    missing_tools = [name for name in tool_names if name not in retrieved_tool_names]
    
    if missing_tools:
        logger.warning(f"[RESEARCH_AGENT] Missing tools: {missing_tools}")
    logger.info(f"[RESEARCH_AGENT] Successfully constructed with {len(tools)} tools: {retrieved_tool_names}")
    
    return PydanticRagAgent(llm_provider, agent_config, tools)
