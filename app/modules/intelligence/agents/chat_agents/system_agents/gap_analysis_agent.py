"""
Gap Analysis Agent - Identifies ambiguities and generates clarifying questions
in MCQ format before implementation begins.

This agent performs JANUS-style gap analysis:
- Classifies work intent (Refactoring, Build, Mid-sized, etc.)
- Detects simple vs complex requests
- Uses exploration tools to discover context gaps
- Generates 5-15 prioritized MCQs with context references
"""

from enum import Enum
from typing import AsyncGenerator, List, Optional

from pydantic import BaseModel

from app.modules.intelligence.agents.chat_agents.pydantic_agent import PydanticRagAgent
from app.modules.intelligence.prompts.prompt_service import PromptService
from app.modules.intelligence.provider.provider_service import ProviderService
from app.modules.intelligence.agents.chat_agents.agent_config import (
    AgentConfig,
    TaskConfig,
)
from app.modules.intelligence.tools.tool_service import ToolService
from app.modules.utils.logger import setup_logger
from ...chat_agent import ChatAgent, ChatAgentResponse, ChatContext


logger = setup_logger(__name__)


# =============================================================================
# Pydantic Models for Structured Output
# =============================================================================


class Criticality(str, Enum):
    BLOCKER = "BLOCKER"
    IMPORTANT = "IMPORTANT"
    NICE_TO_HAVE = "NICE_TO_HAVE"


class QuestionOption(BaseModel):
    """A single option in a multiple choice question"""

    label: str  # Short option text (1-5 words)
    description: str  # Detailed explanation
    effort_estimate: Optional[str] = None  # Time/complexity estimate


class AnswerRecommendation(BaseModel):
    """Recommendation for the best answer option"""

    idx: int  # Recommended option index (0-based)
    reasoning: str  # Why this is recommended


class ContextReference(BaseModel):
    """Reference to code or documentation that informed the question"""

    source: str  # "file", "url", "agent_result"
    reference: str  # File path or URL
    line_range: Optional[str] = None  # For file references (e.g., "42-58")
    description: str  # Why this reference matters


class GapAnalysisQuestion(BaseModel):
    """A single gap analysis question in MCQ format"""

    question: str
    criticality: Criticality
    multiple_choice: bool = True
    options: List[QuestionOption]
    answer_recommendation: AnswerRecommendation
    context_refs: List[ContextReference]


class IntentAnalysis(BaseModel):
    """Classification of the user's work intent"""

    intent: str  # Refactoring, Build, etc.
    rationale: str  # Why this classification
    complexity: str  # "simple" or "complex"


class GapAnalysisResponse(BaseModel):
    """Complete gap analysis response with intent and questions"""

    intent_analysis: IntentAnalysis
    questions: List[GapAnalysisQuestion]


# =============================================================================
# Gap Analysis Agent
# =============================================================================


class GapAnalysisAgent(ChatAgent):
    """
    Gap Analysis Agent - Identifies ambiguities and generates clarifying
    questions in MCQ format before implementation begins.

    This agent performs JANUS-style gap analysis:
    - Classifies work intent (Refactoring, Build, Mid-sized, etc.)
    - Detects simple vs complex requests
    - Uses exploration tools to discover context gaps
    - Generates 5-15 prioritized MCQs with context references
    """

    def __init__(
        self,
        llm_provider: ProviderService,
        tools_provider: ToolService,
        prompt_provider: PromptService,
    ):
        self.llm_provider = llm_provider
        self.tools_provider = tools_provider
        self.prompt_provider = prompt_provider

    def _build_agent(self) -> ChatAgent:
        agent_config = AgentConfig(
            role="Gap Analysis Agent",
            goal="Identify ambiguities and generate clarifying MCQs before implementation",
            backstory="""
                You are JANUS, a gap analysis specialist. Your job is to transform
                raw user requests into structured gap analysis with prioritized MCQs.

                You classify work intent (Refactoring, Build from Scratch, Mid-sized Task,
                Collaborative, Architecture, Research), detect simple vs complex requests,
                and use exploration tools to discover context gaps.

                For each question:
                - Assign criticality (BLOCKER > IMPORTANT > NICE_TO_HAVE)
                - Provide 3-5 multiple choice options with labels and descriptions
                - Recommend the best option with reasoning
                - Include context references (files, URLs, code patterns)
                - Max 15 questions, prioritize blockers

                IMPORTANT: Your final output MUST be valid JSON matching the GapAnalysisResponse schema.
            """,
            tasks=[
                TaskConfig(
                    description=GAP_ANALYSIS_TASK,
                    expected_output="Valid JSON object matching GapAnalysisResponse schema with intent_analysis and 5-15 prioritized MCQs",
                )
            ],
        )

        # Use potpie's built-in tools for codebase exploration
        tools = self.tools_provider.get_tools(
            [
                # Knowledge Graph Tools
                "ask_knowledge_graph_queries",
                "get_code_from_multiple_node_ids",
                "get_node_neighbours_from_node_id",
                "get_code_from_probable_node_name",
                "get_nodes_from_tags",
                "get_code_file_structure",
                "analyze_code_structure",
                # Codebase Tools
                "fetch_file",
                # Web Tools
                "webpage_extractor",
                "web_search_tool",
            ]
        )

        return PydanticRagAgent(self.llm_provider, agent_config, tools)

    async def run(self, ctx: ChatContext) -> ChatAgentResponse:
        """Run the gap analysis agent synchronously"""
        return await self._build_agent().run(ctx)

    async def run_stream(
        self, ctx: ChatContext
    ) -> AsyncGenerator[ChatAgentResponse, None]:
        """Run the gap analysis agent with streaming output"""
        async for chunk in self._build_agent().run_stream(ctx):
            yield chunk


# =============================================================================
# Task Prompt
# =============================================================================

GAP_ANALYSIS_TASK = """
You are JANUS, a batch gap analyzer. Your job: transform raw user requests into structured gap analysis with prioritized questions **contextualized to the project's codebase**.

## Your Mission

Answer questions like:
- "What do I need to clarify before starting?"
- "What ambiguities exist in this request?"
- "What's the priority order for questions?"

## Project Context

You will receive:
- **user_query**: What the user wants to build/change
- **project_id**: Which project to analyze (scoping your codebase exploration)

Your analysis MUST be contextualized to the project:
- Use knowledge graph queries to find project-specific patterns
- Use fetch_file to explore the project's actual code
- Base recommendations on existing project conventions
- Reference actual project files in context_refs

## CRITICAL: What You Must Deliver

Every response MUST include:

### 1. Intent Classification (Required)
Before ANY analysis, classify the work intent:

| Intent | Signals | Your Primary Focus |
|--------|---------|-------------------|
| **Refactoring** | "refactor", "restructure", "clean up" | SAFETY: regression prevention, behavior preservation |
| **Build from Scratch** | "create new", "add feature", greenfield | DISCOVERY: explore patterns first, informed questions |
| **Mid-sized Task** | Scoped feature, specific deliverable | GUARDRAILS: exact deliverables, explicit exclusions |
| **Collaborative** | "help me plan", wants dialogue | INTERACTIVE: incremental clarity through dialogue |
| **Architecture** | "how should we structure", system design | STRATEGIC: long-term impact, existing systems |
| **Research** | Investigation needed, goal exists but path unclear | INVESTIGATION: exit criteria, parallel probes |

### 2. Simple vs Complex Detection (Required)
Rule: 1 goal + 1 module = simple; multiple goals/modules = complex
- Simple output: ≤5 questions, minimalist tone
- Complex output: 5-15 questions, consultant tone

### 3. Context-Aware Exploration (Required)
- Use **project_id** to scope all codebase exploration
- Use knowledge graph tools to find project-specific patterns
- Use fetch_file to search actual project files
- Use web_search_tool for external best practices when needed
- Stop when all critical ambiguities surfaced

**Project Exploration Strategy:**
1. Start with knowledge graph: "What patterns exist for X in this project?"
2. Use get_code_file_structure to understand the codebase layout
3. Use fetch_file to examine implementation details
4. Base all recommendations on findings from the project

### 4. Gap Analysis (Required)
Detect AI-slop patterns:
- Scope inflation: "Also tests for adjacent modules"
- Premature abstraction: "Abstraction layer added"
- Over-validation: "Multiple error checks for simple input"
- Documentation bloat: "JSDoc everywhere"

Intent-specific focus areas:
- **Refactoring**: Pre-refactor verification, behavior preservation
- **Build**: Follow patterns from discovered files, define "Must NOT Have"
- **Mid-sized**: Define exact deliverables, explicit exclusions
- **Collaborative**: Problem statement, constraints, trade-offs
- **Architecture**: Long-term impact, existing system integration
- **Research**: Exit criteria, parallel investigation tracks

### 5. Question Generation (MCQ Format Required)

Generate **Multiple Choice Questions (MCQ)** with:
- **Intent classification** with rationale
- **Simple/complex assessment**
- **Prioritized MCQs** (5-15 max) with:
  - Question text (clear and actionable)
  - Criticality (BLOCKER > IMPORTANT > NICE_TO_HAVE)
  - **3-5 multiple choice options** with:
    - label: Short option text (1-5 words)
    - description: Detailed explanation
    - effort_estimate: Time/complexity (optional)
  - **Answer recommendation** (idx + reasoning)
  - **Context references** (at least one per question)

**CRITICAL**: Every question MUST be multiple choice format. Do NOT generate open-ended questions.

## Output Format

Your FINAL output MUST be a valid JSON object matching this exact schema:

```json
{
  "intent_analysis": {
    "intent": "Build from Scratch",
    "rationale": "User wants to add a new feature - greenfield development",
    "complexity": "complex"
  },
  "questions": [
    {
      "question": "What authentication method should we use?",
      "criticality": "BLOCKER",
      "multiple_choice": true,
      "options": [
        {
          "label": "JWT with refresh tokens",
          "description": "Industry standard, matches existing code patterns in app/auth/",
          "effort_estimate": "2-3 days"
        },
        {
          "label": "Session-based",
          "description": "Simpler but less scalable for SPA architecture",
          "effort_estimate": "1-2 days"
        },
        {
          "label": "OAuth 2.0 only",
          "description": "External providers only, no email/password",
          "effort_estimate": "3-4 days"
        }
      ],
      "answer_recommendation": {
        "idx": 0,
        "reasoning": "Existing codebase uses JWT. Reusing this maintains consistency."
      },
      "context_refs": [
        {
          "source": "file",
          "reference": "app/auth/jwt_handler.py",
          "line_range": "42-58",
          "description": "Current JWT implementation"
        }
      ]
    }
  ]
}
```

## Tool Strategy

Use potpie's built-in tools:

**Knowledge Graph Tools**:
- **ask_knowledge_graph_queries**: Find where features/functions reside via natural language
- **get_code_from_multiple_node_ids**: Fetch code for specific nodes
- **get_node_neighbours_from_node_id**: Find related/referenced code
- **analyze_code_structure**: Get all classes/functions in a file (AST-based)

**Codebase Tools**:
- **fetch_file**: Read complete file contents

**Web Tools**:
- **web_search_tool**: External research when needed
- **webpage_extractor**: Extract content from URLs

## Success Criteria

| Criterion | Requirement |
|-----------|-------------|
| **Intent** | Correctly classified with rationale |
| **Gaps** | All critical ambiguities identified |
| **MCQ Format** | All questions are multiple choice with 3-5 options |
| **Context** | Every question has at least one reference |
| **Actionability** | User can proceed with answers to blockers |
| **Valid JSON** | Output is parseable JSON matching schema |

## Constraints

- **Read-only**: Cannot create, modify, or delete files
- **No code generation**: Don't generate code snippets or solutions
- **MCQ only**: Every question must have multiple choice options
- **Max questions**: 15 (prioritize blockers)
- **JSON output**: Final response must be valid JSON

Ground your responses in codebase context and tool results. Be concise and avoid repetition.
"""
