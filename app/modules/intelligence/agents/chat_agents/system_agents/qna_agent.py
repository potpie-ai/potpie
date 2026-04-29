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
from typing import AsyncGenerator, Optional, TYPE_CHECKING, FrozenSet
from app.modules.utils.logger import setup_logger

if TYPE_CHECKING:
    from app.modules.intelligence.tools.registry.resolver import ToolResolver

logger = setup_logger(__name__)

REPO_GUIDANCE_FILES: FrozenSet[str] = frozenset({
    "AGENTS.md", "agents.md", "skills.md", "SKILLS.md",
    ".github/copilot-instructions.md", ".cursor/rules",
})


class QnAAgent(ChatAgent):
    def __init__(
        self,
        llm_provider: ProviderService,
        tools_provider: ToolService,
        prompt_provider: PromptService,
        tool_resolver: Optional["ToolResolver"] = None,
    ):
        self.llm_provider = llm_provider
        self.tools_provider = tools_provider
        self.prompt_provider = prompt_provider
        self.tool_resolver = tool_resolver

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
                "fetch_file",
                "fetch_files_batch",
                "analyze_code_structure",
                "search_colgrep",
                "check_colgrep_health",
                "bash_command",
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
- Use search_colgrep as your PRIMARY discovery tool for finding relevant code
- Use bash_command with rg for targeted text search, scoped to ColGREP results
- Use fetch_file, fetch_files_batch for reading actual source code
- Use get_code_file_structure and analyze_code_structure for structural understanding
- Prefer rg over grep in all bash_command searches
</tooling_rules>

<delegation_policy>
CONSERVATIVE DELEGATION for Q&A:
- Do your own ColGREP searches and initial file reads FIRST
- Only delegate to subagent AFTER you have identified specific candidate paths
- Delegate for: deep-dive investigation of a specific subsystem, parallel independent branches
- Do NOT delegate the initial discovery phase — that's YOUR job as supervisor
</delegation_policy>

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
                    tool_resolver=self.tool_resolver,
                    supervisor_allow_list="qna_supervisor",
                    execute_allow_list="qna_execute",
                    read_only=True,
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
        ctx = await self._seed_repo_guidance(ctx)
        return ctx

    async def _seed_top_level_structure(self, ctx: ChatContext) -> ChatContext:
        if "Top-level code map" in ctx.additional_context:
            return ctx

        try:
            file_structure = await self.tools_provider.file_structure_tool.fetch_repo_structure(
                project_id=ctx.project_id, path=None
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

    async def _seed_repo_guidance(self, ctx: ChatContext) -> ChatContext:
        if "Repo guidance" in ctx.additional_context:
            return ctx

        # Scan the already-seeded structure for guidance files
        found = [f for f in REPO_GUIDANCE_FILES if f in ctx.additional_context]
        if found:
            prefix = "" if not ctx.additional_context else "\n"
            ctx.additional_context += (
                f"{prefix}Repo guidance detected — read these files FIRST before investigating: "
                + ", ".join(sorted(found))
            )
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
        ctx = await self._enriched_context(ctx)
        return await self._build_agent(ctx).run(ctx)

    async def run_stream(
        self, ctx: ChatContext
    ) -> AsyncGenerator[ChatAgentResponse, None]:
        ctx = await self._enriched_context(ctx)
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
PHASE 3: RETRIEVAL PIPELINE (ColGREP-first)
================================================================================

<retrieval_sequence>
Execute in this order. ColGREP is your PRIMARY discovery tool.

STEP 1 - ColGREP SEMANTIC SEARCH:
  Tool: search_colgrep(query=...)
  Purpose: Find semantically relevant files/symbols across the indexed repository
  Rule: ALWAYS start here. Formulate a clear, keyword-rich query.
  If ColGREP is unhealthy or returns no results, fall back to STEP 2.

STEP 2 - TARGETED TEXT SEARCH (scoped by ColGREP results):
  Tool: bash_command
  Command: rg "pattern" [paths from ColGREP results]
  Rule: Use rg (ripgrep), never grep. Scope searches to paths ColGREP identified.
  If ColGREP returned nothing, do a broader rg scan.

STEP 3 - CODE RETRIEVAL:
  Tool: fetch_file / fetch_files_batch
  Purpose: Read actual source with line ranges after narrowing scope via Steps 1-2

STEP 4 - STRUCTURAL ANALYSIS (optional):
  Tool: analyze_code_structure
  Purpose: Summarize classes/functions in specific files before deeper reads

STEP 5 - STRUCTURE MAPPING (if path still unknown):
  Tool: get_code_file_structure(path=relevant_directory)
  Purpose: Browse directory structure when ColGREP + rg didn't locate the target
</retrieval_sequence>

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
