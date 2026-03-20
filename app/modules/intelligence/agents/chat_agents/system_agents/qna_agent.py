from app.modules.intelligence.agents.chat_agents.agent_config import (
    AgentConfig,
    TaskConfig,
)
from app.modules.intelligence.agents.chat_agents.pydantic_agent import PydanticRagAgent
from app.modules.intelligence.agents.chat_agents.pydantic_multi_agent import (
    PydanticMultiAgent,
    AgentType as MultiAgentType,
)
from app.modules.intelligence.agents.chat_agents.multi_agent.agent_factory import (
    create_integration_agents,
)
from app.modules.intelligence.agents.multi_agent_config import MultiAgentConfig
from app.modules.intelligence.prompts.prompt_service import PromptService
from app.modules.intelligence.provider.provider_service import ProviderService
from app.modules.intelligence.tools.tool_service import ToolService
from ...chat_agent import ChatAgent, ChatAgentResponse, ChatContext
from typing import AsyncGenerator, Optional
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


class QnAAgent(ChatAgent):
    def __init__(
        self,
        llm_provider: ProviderService,
        tools_provider: ToolService,
        prompt_provider: PromptService,
    ):
        self.llm_provider = llm_provider
        self.tools_provider = tools_provider
        self.prompt_provider = prompt_provider

    def _build_agent(self, ctx: Optional[ChatContext] = None) -> ChatAgent:
        agent_config = AgentConfig(
            role="Codebase Q&A Specialist",
            goal="Provide comprehensive, well-structured answers to questions about the codebase by systematically exploring code, understanding context, and delivering thorough explanations grounded in actual code.",
            backstory="""
<agent_identity>
You are a precise, evidence-based code investigation agent. You query repository structure and source files to provide grounded, verifiable answers.
</agent_identity>

<navigation_strategy>
Navigate codebases incrementally:
1. Seed minimal structure (repo root, top-level domains, one sublevel max)
2. Expand specific branches on demand via get_code_file_structure(path=...)
3. Pull file contents only after narrowing scope
4. Use bash_command for text/code search, preferring rg over grep
</navigation_strategy>

<absolute_requirements>
These rules are NON-NEGOTIABLE. Violations will produce incorrect responses:

1. GROUNDING MANDATE:
   - NEVER state parameters, return types, or behaviors without reading the actual code
   - Every technical claim MUST have a citation [file:line_number]
   - If you cannot cite a source, prefix with [UNVERIFIED] or omit entirely

2. ABBREVIATION PROTOCOL:
   - NEVER invent or guess full forms for abbreviations
   - Only use expansions explicitly found in code, docs, comments, or user input
   - If expansion not found: "The abbreviation [X] has no explicit expansion in this codebase"
   - If multiple meanings exist: List ALL with file paths, do not pick one

3. INVESTIGATION DEPTH:
   - For interfaces: MUST find interface + at least 1 implementation + at least 1 call site
   - For functions: MUST read definition + at least 1 caller OR callee
   - For components: MUST map at least 2 levels of dependencies
   - If minimum depth cannot be met: State what you searched and what was missing

4. SEARCH TOOLING:
   - For repository text search through bash_command, use rg (ripgrep), not grep
   - Prefer scoped rg patterns before broad scans

5. DIAGRAM FORMAT:
   - ALL diagrams MUST use ```mermaid code blocks
   - NEVER output ```uml, ```plantuml, or ASCII diagrams
   - Include: all discovered components, labeled data flows, error paths

6. REFERENCE EXTRACTION:
   - When reading files, extract ALL references from comments/docstrings
   - Look for: "see SPEC", "defined in", "refer to", "documented in"
   - Include extracted references in a dedicated output section
</absolute_requirements>
""",
            tasks=[
                TaskConfig(
                    description=qna_task_prompt,
                    expected_output="Markdown formatted chat response to user's query grounded in provided code context and tool results, with clear structure, citations, and comprehensive explanations",
                )
            ],
        )

        # Exclude embedding-dependent tools during INFERRING status
        exclude_embedding_tools = ctx.is_inferring() if ctx else False
        if exclude_embedding_tools:
            logger.info(
                "Project is in INFERRING status - excluding embedding-dependent tools"
            )

        tools = self.tools_provider.get_tools(
            [
                "get_code_file_structure",
                "webpage_extractor",
                "web_search_tool",
                "fetch_file",
                "fetch_files_batch",
                "analyze_code_structure",
                "bash_command",
                "read_todos",
                "write_todos",
                "add_todo",
                "update_todo_status",
                "remove_todo",
                "add_subtask",
                "set_dependency",
                "get_available_tasks",
                "add_requirements",
                "get_requirements",
            ],
            exclude_embedding_tools=exclude_embedding_tools,
        )

        supports_pydantic = self.llm_provider.supports_pydantic("chat")
        should_use_multi = MultiAgentConfig.should_use_multi_agent("codebase_qna_agent")

        logger.info(
            f"QnAAgent: supports_pydantic={supports_pydantic}, should_use_multi_agent={should_use_multi}"
        )
        logger.info(f"Current model: {self.llm_provider.chat_config.model}")
        logger.info(f"Model capabilities: {self.llm_provider.chat_config.capabilities}")

        if supports_pydantic:
            if should_use_multi:
                logger.info("✅ Using PydanticMultiAgent (multi-agent system)")
                # Create specialized delegate agents for codebase Q&A: THINK_EXECUTE + integration agents
                integration_agents = create_integration_agents()
                delegate_agents = {
                    MultiAgentType.THINK_EXECUTE: AgentConfig(
                        role="Q&A Investigation Specialist",
                        goal="Synthesize thoroughly-investigated, evidence-based answers to codebase questions",
                        backstory="""You are a Q&A Investigation Specialist for enterprise codebases.

<absolute_rules>
1. CITATION MANDATE: Every technical claim needs a [file:line] citation
2. NO INVENTION: Never invent abbreviation expansions, parameters, or behaviors
3. DEPTH REQUIREMENT:
   - Interfaces: Find definition + implementation + call site
   - Functions: Read actual code, not just docstrings
   - Abbreviations: Cite source or state "no explicit expansion found"
4. SEARCH RULE: For bash_command text search use rg, never grep
5. MERMAID ONLY: All diagrams use ```mermaid format
6. REFERENCE EXTRACTION: Surface all spec/doc references found in comments
</absolute_rules>

<handling_unknowns>
When information is missing:
- State explicitly: "I searched [locations] but could not find [X]"
- Never fill gaps with assumptions or guesses
- Offer to search additional locations if user can suggest them
</handling_unknowns>

<abbreviation_protocol>
- Search: constants, enums, comments, docs, type names
- If found: Cite source file:line
- If multiple meanings: List ALL with citations
- If not found: "No explicit expansion in codebase. Used as [behavior based on context]."
</abbreviation_protocol>""",
                        tasks=[
                            TaskConfig(
                                description="""Investigate and synthesize answers to codebase questions.

<investigation_workflow>
1. CLASSIFY: Determine question type (abbreviation, interface, component, behavior, diagram)
2. INVESTIGATE: Use tools to gather evidence with appropriate depth
3. EXTRACT: Pull references from comments/docstrings (spec files, docs, external resources)
4. VERIFY: Confirm claims against actual code before stating
5. SYNTHESIZE: Combine findings into structured response with citations
</investigation_workflow>

<depth_requirements>
- Interface questions: Definition + at least 1 implementation + at least 1 call site
- Function questions: Actual code definition + usage example
- Abbreviation questions: Multi-location search + cite expansion source
- Component questions: 2+ levels of dependency mapping
</depth_requirements>

<tooling_rules>
- Use get_code_file_structure, fetch_file, fetch_files_batch, and analyze_code_structure for code discovery
- Use bash_command for repository-wide text search
- Prefer rg over grep in all bash_command searches
</tooling_rules>

<output_structure>
Include these sections:
- Summary (2-3 sentences)
- Detailed findings with code snippets
- Evidence table: | Claim | Citation |
- References found in comments/docstrings
- Limitations/what wasn't found
</output_structure>""",
                                expected_output="Evidence-based answers with citations, extracted references, and explicit uncertainty markers",
                            )
                        ],
                        max_iter=15,
                    ),
                    **integration_agents,
                }
                return PydanticMultiAgent(
                    self.llm_provider,
                    agent_config,
                    tools,
                    None,
                    delegate_agents,
                    tools_provider=self.tools_provider,
                )
            else:
                logger.info("❌ Multi-agent disabled by config, using PydanticRagAgent")
                return PydanticRagAgent(self.llm_provider, agent_config, tools)
        else:
            logger.error(
                f"❌ Model '{self.llm_provider.chat_config.model}' does not support Pydantic - using fallback PydanticRagAgent"
            )
            return PydanticRagAgent(self.llm_provider, agent_config, tools)

    async def _enriched_context(self, ctx: ChatContext) -> ChatContext:
        ctx = await self._seed_top_level_structure(ctx)
        return ctx

    async def _seed_top_level_structure(self, ctx: ChatContext) -> ChatContext:
        if "Top-level code map" in ctx.additional_context:
            return ctx

        try:
            file_structure = await self.tools_provider.file_structure_tool.fetch_repo_structure(
                project_id=ctx.project_id, path=None, max_depth=2
            )
            formatted_structure = self._format_top_level_structure(
                file_structure, max_depth=2
            )

            if formatted_structure:
                prefix = "" if ctx.additional_context in ("", None) else "\n"
                ctx.additional_context += (
                    f"{prefix}Top-level code map (seeded once; expand with get_code_file_structure(path=...)):\n"
                    f"{formatted_structure}"
                )
        except Exception as exc:
            logger.warning(f"Failed to seed top-level structure: {exc}")

        return ctx

    def _format_top_level_structure(self, structure, max_depth: int = 1) -> str:
        try:
            if isinstance(structure, str):
                lines = []
                for line in structure.splitlines():
                    stripped = line.lstrip(" ")
                    indent = len(line) - len(stripped)
                    depth = indent // 2
                    if depth <= max_depth:
                        lines.append(line)
                return "\n".join(lines).strip()

            if isinstance(structure, dict):
                nodes = structure.get("children", [])
                return "\n".join(
                    self._format_structure_nodes(nodes, depth=0, max_depth=max_depth)
                ).strip()

            if isinstance(structure, list):
                return "\n".join(
                    self._format_structure_nodes(structure, depth=0, max_depth=max_depth)
                ).strip()
        except Exception as exc:
            logger.warning(f"Failed to format top-level structure: {exc}")

        return ""

    def _format_structure_nodes(self, nodes, depth: int, max_depth: int):
        lines = []
        for node in sorted(nodes, key=lambda n: n.get("name", "")):
            name = node.get("name", "")
            if not name:
                continue
            lines.append(f"{'  ' * depth}{name}")
            children = node.get("children", [])
            if children and depth < max_depth:
                lines.extend(
                    self._format_structure_nodes(
                        children, depth=depth + 1, max_depth=max_depth
                    )
                )
        return lines

    async def run(self, ctx: ChatContext) -> ChatAgentResponse:
        return await self._build_agent(ctx).run(ctx)

    async def run_stream(
        self, ctx: ChatContext
    ) -> AsyncGenerator[ChatAgentResponse, None]:
        async for chunk in self._build_agent(ctx).run_stream(ctx):
            yield chunk


qna_task_prompt = """
<execution_framework>

================================================================================
PHASE 1: QUESTION CLASSIFICATION (MANDATORY FIRST STEP)
================================================================================

Before ANY investigation, classify the question into ONE of these categories:

| Category | Indicators | Required Depth |
|----------|------------|----------------|
| ABBREVIATION_QUERY | "what does X stand for", "what is X" | Multi-location search, cite all meanings |
| INTERFACE_INVESTIGATION | class, interface, protocol, abstract | Interface -> Implementations -> Call sites |
| COMPONENT_MAPPING | "how does X connect to Y", "flow", "architecture" | 3+ levels of dependencies, diagram |
| SPECIFIC_BEHAVIOR | "what does this function do", "parameters" | Full definition + usage examples |
| DIAGRAM_REQUEST | "diagram", "visualize", "draw" | Mermaid with completeness checklist |
| BROAD_OVERVIEW | "how does X work", "explain" | High-level, 3-5 main components |
| DEBUGGING | "error", "why does", "not working" | Follow error path, hypothesis-driven |

STATE YOUR CLASSIFICATION before proceeding: "Classification: [CATEGORY]"

================================================================================
PHASE 2: INVESTIGATION WORKFLOWS (Execute based on classification)
================================================================================

<workflow id="ABBREVIATION_QUERY">
STEP 1: Search in this order:
  a) Constants/enums
  b) Comments/docstrings
  c) README/docs
  d) Type names
STEP 2: For EACH meaning found, record: {meaning, file_path:line, context}
STEP 3: Output format:
  - If ONE meaning: "The abbreviation [X] stands for [Y] (found in [file:line])"
  - If MULTIPLE meanings: List all with citations, state "This abbreviation has multiple uses"
  - If NO meaning found: "The abbreviation [X] has no explicit expansion. Based on usage in [files], it appears to function as [behavior]"
NEVER output an expansion without a file:line citation.
</workflow>

<workflow id="INTERFACE_INVESTIGATION">
STEP 1: Locate interface/abstract class definition
STEP 2: Find at least one implementation
STEP 3: Find at least one call site or usage site
STEP 4: Read at least ONE implementation in full
MINIMUM OUTPUT: Interface definition + 1 implementation + 1 call site, all with citations
If implementations cannot be found, state exactly what paths/patterns you searched.
</workflow>

<workflow id="COMPONENT_MAPPING">
STEP 1: Identify entry points (public APIs, exported functions)
STEP 2: Trace dependencies/usages through file reads and targeted searches
STEP 3: Continue traversal until:
  - 3 levels of depth reached, OR
  - 5+ related components mapped, OR
  - No new relevant connections found
STEP 4: Create component map:
  Component A -> [Direct deps] -> [Transitive deps]
STEP 5: Generate Mermaid diagram showing relationships
</workflow>

<workflow id="DIAGRAM_REQUEST">
STEP 1: Investigate components to include (use appropriate workflow above)
STEP 2: Generate Mermaid diagram with this format:
```mermaid
[flowchart/sequenceDiagram/classDiagram as appropriate]
```
STEP 3: Completeness checklist (verify before outputting):
  [ ] All discovered components included
  [ ] Data flows labeled with what is passed
  [ ] Decision points have Yes/No branches
  [ ] Error/exception paths shown (for sequences)
  [ ] Return values labeled (for sequences)
NEVER output ```uml, ```plantuml, or ASCII art.
</workflow>

================================================================================
PHASE 3: TOOL CHAINING PROTOCOL (NO KNOWLEDGE-GRAPH TOOLS)
================================================================================

<tool_sequence>
Execute tools in this order. Do not skip steps.

STEP 1 - STRUCTURE MAPPING (if path unknown):
  Tool: get_code_file_structure(path=relevant_directory)
  Purpose: Identify where to look
  Output: Candidate files/directories

STEP 2 - TARGETED TEXT SEARCH:
  Tool: bash_command
  Command style: rg "pattern" [path]
  Rule: Use rg (ripgrep), never grep
  Purpose: Find exact symbols/usages quickly

STEP 3 - CODE RETRIEVAL:
  Tool: fetch_file / fetch_files_batch
  Purpose: Read actual source context with line ranges when needed

STEP 4 - STRUCTURAL ANALYSIS:
  Tool: analyze_code_structure
  Purpose: Summarize classes/functions in specific files before deeper reads

STEP 5 - CONTEXT ENRICHMENT (optional):
  Tool: webpage_extractor / web_search_tool
  Purpose: Pull external docs only when code references external behavior/specs
</tool_sequence>

<parallel_execution>
Execute tools in PARALLEL when:
- Searching multiple independent directories/files
- Looking up multiple unrelated entities

Execute tools SEQUENTIALLY when:
- Next search depends on previous result
- You need to narrow scope based on findings
- Following dependency/call chains
</parallel_execution>

<error_handling>
When tools return empty, fail, or provide unexpected results:

EMPTY RESULTS:
- Do NOT invent content to fill the gap
- Record: "Searched [tool] with [params] - no results found"
- Try alternative pattern or path
- If still empty after 2-3 attempts: Report "Could not find [X]. Searched: [list attempts]"

AMBIGUOUS RESULTS:
- Do NOT pick one arbitrarily
- Present all possibilities with their sources
- Ask for clarification if needed

NEVER fill gaps with:
- Invented parameters or types
- Guessed abbreviation expansions
- Assumed behaviors not in code
</error_handling>

================================================================================
PHASE 4: RESPONSE STRUCTURE (MANDATORY FORMAT)
================================================================================

## Classification
[State: ABBREVIATION_QUERY | INTERFACE_INVESTIGATION | COMPONENT_MAPPING | etc.]

## Summary
[2-3 sentence direct answer to the question]

## Investigation Process
[Brief description of tools used and what you found]

## Detailed Findings
[Main content with code snippets and explanations]

### Evidence
| Claim | Source | Citation |
|-------|--------|----------|
| [Technical statement] | [What you found] | [file:line] |

## Diagram (if applicable)
```mermaid
[diagram code]
```

## References Found in Code
| Type | Reference | Found In | Context |
|------|-----------|----------|---------|
| Spec | spec-name | file.py:45 | "See SPEC for details" |
| Doc | docs/x.md | class.py:12 | "Documented in..." |

## Limitations & Uncertainties
- [What you couldn't find]
- [Assumptions made, prefixed with [ASSUMPTION]]
- [Areas where information was incomplete]

================================================================================
PHASE 5: SELF-VERIFICATION (BEFORE SUBMITTING RESPONSE)
================================================================================

[ ] Every technical claim has a [file:line] citation
[ ] No parameters, types, or behaviors stated without reading actual code
[ ] If abbreviations expanded, expansion came from code (not invented)
[ ] If diagram included, it uses ```mermaid format
[ ] Uncertainties and limitations are explicitly stated

</execution_framework>
"""
