"""Main SpecGenAgent orchestrator for the 7-step specification generation workflow."""
import json
import re
from typing import AsyncGenerator, Optional, TypeVar, Type, List
from pydantic import BaseModel
from app.modules.intelligence.agents.chat_agents.agent_config import AgentConfig, TaskConfig
from app.modules.intelligence.agents.chat_agents.pydantic_agent import PydanticRagAgent
from app.modules.intelligence.prompts.prompt_service import PromptService
from app.modules.intelligence.provider.provider_service import ProviderService
from app.modules.intelligence.tools.tool_service import ToolService
from app.modules.intelligence.agents.chat_agent import ChatAgent, ChatAgentResponse, ChatContext
from app.modules.utils.logger import setup_logger

T = TypeVar('T', bound=BaseModel)

from .spec_models import (
    ResearchFindings,
    RequirementsOutputCore,
    RequirementEnrichmentOutput,
    RequirementsOutput,
    ArchitectureOutput,
    TechnicalDesignOutput,
    ValidationReport,
    Specification,
    FunctionalRequirement,
    NonFunctionalRequirement,
    RequirementEnrichment,
    RequirementAppendix,
)
from .research_agent import create_research_agent
from .requirements_agent import create_requirements_agent
from .enrichment_agent import create_enrichment_agent
from .architecture_agent import create_architecture_agent
from .technical_design_agent import create_technical_design_agent
from .validator_agent import create_validator_agent
from .assembler_agent import create_assembler_agent

logger = setup_logger(__name__)


def parse_agent_output(result: any, model_class: Type[T]) -> T:
    """
    Parse agent output into structured model.
    
    Handles both old (.data) and new (.output) pydantic_ai result formats.
    Extracts JSON from text output if needed.
    Normalizes common data format issues (e.g., success_metrics dictionaries to strings).
    
    Includes retry logic: on ValidationError, retries once with additional normalization.
    """
    from pydantic import ValidationError
    
    # Try to get output - check both .data and .output attributes
    raw_output = None
    if hasattr(result, 'data'):
        raw_output = result.data
    elif hasattr(result, 'output'):
        raw_output = result.output
    else:
        raise AttributeError(f"Agent result missing expected attribute. Available: {dir(result)}")
    
    # If already a model instance, return it
    if isinstance(raw_output, model_class):
        return raw_output
    
    # Helper function to attempt validation with normalization
    def _try_validate(data: any, is_retry: bool = False) -> T:
        """Attempt validation with normalization, returns validated model or raises."""
        if isinstance(data, dict):
            normalized_dict = _normalize_output_dict(data, model_class)
            try:
                return model_class.model_validate(normalized_dict)
            except ValidationError as e:
                if is_retry:
                    # Already retried once, re-raise
                    logger.error(f"[SPEC_GEN] Validation failed after retry for {model_class.__name__}: {e}")
                    raise
                else:
                    # First attempt failed, log and retry with additional normalization
                    logger.warning(f"[SPEC_GEN] Validation failed for {model_class.__name__}, retrying with additional normalization: {e}")
                    # Re-normalize (may catch additional issues)
                    try:
                        normalized_dict = _normalize_output_dict(normalized_dict, model_class)
                        return model_class.model_validate(normalized_dict)
                    except ValidationError as retry_error:
                        logger.error(f"[SPEC_GEN] Validation failed after retry for {model_class.__name__}: {retry_error}")
                        raise
        else:
            # For non-dict data, try direct validation
            try:
                return model_class.model_validate(data)
            except ValidationError as e:
                if is_retry:
                    logger.error(f"[SPEC_GEN] Validation failed after retry for {model_class.__name__}: {e}")
                    raise
                else:
                    logger.warning(f"[SPEC_GEN] Validation failed for {model_class.__name__}, retrying: {e}")
                    # Try converting to dict if possible
                    if hasattr(data, '__dict__'):
                        try:
                            return _try_validate(data.__dict__, is_retry=True)
                        except ValidationError:
                            raise
                    elif isinstance(data, (list, tuple)) and len(data) > 0 and isinstance(data[0], dict):
                        # Might be a list wrapped in something
                        try:
                            return _try_validate(data[0] if len(data) == 1 else data, is_retry=True)
                        except ValidationError:
                            raise
                    else:
                        raise
    
    # If it's a dict, normalize and validate (with retry)
    if isinstance(raw_output, dict):
        return _try_validate(raw_output)
    
    # If it's a string, try to extract JSON and parse
    if isinstance(raw_output, str):
        # Try to extract JSON from markdown code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw_output, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            # Try parsing the whole string as JSON
            json_str = raw_output.strip()
        
        try:
            json_dict = json.loads(json_str)
            return _try_validate(json_dict)
        except json.JSONDecodeError as e:
            # Attempt recovery from invalid escape sequences
            try:
                logger.warning(f"[SPEC_GEN] JSON parsing failed, attempting recovery from escape sequences: {e}")
                # Try unicode_escape decoding
                repaired = json_str.encode("utf-8").decode("unicode_escape")
                json_dict = json.loads(repaired)
                logger.info("[SPEC_GEN] JSON parsing recovery successful using unicode_escape")
                return _try_validate(json_dict)
            except Exception:
                # Final fallback: double escape backslashes
                try:
                    repaired = json_str.replace("\\", "\\\\")
                    json_dict = json.loads(repaired)
                    logger.info("[SPEC_GEN] JSON parsing recovery successful using double escape")
                    return _try_validate(json_dict)
                except Exception:
                    # All recovery attempts failed, raise original error
                    logger.error(f"Failed to parse JSON from agent output after recovery attempts: {e}")
                    logger.debug(f"Raw output (first 500 chars): {raw_output[:500]}")
                    raise ValueError(f"Agent output is not valid JSON: {e}")
    
    # Fallback: try to validate directly (with retry)
    try:
        return _try_validate(raw_output)
    except ValidationError:
        # Already logged in _try_validate
        raise
    except Exception as e:
        logger.error(f"Failed to parse agent output as {model_class.__name__}: {e}")
        raise


def _normalize_evidence(evidence_value: any) -> list:
    """
    Normalize evidence to List[Evidence] format.
    
    Handles:
    - string → convert to single Evidence dict
    - list[str] → convert each string to Evidence dict
    - list[dict] → validate and normalize each dict
    """
    from .spec_models import EvidenceType
    
    if isinstance(evidence_value, str):
        # Single string → convert to Evidence dict
        logger.info(f"[SPEC_GEN] Normalizing evidence from string to Evidence dict")
        return [{
            "type": "agent_result",  # Default type
            "source": "unspecified",
            "relevance": evidence_value
        }]
    elif isinstance(evidence_value, list):
        normalized_evidence = []
        for item in evidence_value:
            if isinstance(item, str):
                # String in list → convert to Evidence dict
                normalized_evidence.append({
                    "type": "agent_result",
                    "source": "unspecified",
                    "relevance": item
                })
            elif isinstance(item, dict):
                # Already a dict → ensure it has required fields
                evidence_dict = {
                    "type": item.get("type", "agent_result"),
                    "source": item.get("source", "unspecified"),
                    "relevance": item.get("relevance", item.get("quote", "Generated evidence"))
                }
                normalized_evidence.append(evidence_dict)
            else:
                # Unknown type → convert to string and create Evidence
                logger.warning(f"[SPEC_GEN] Unknown evidence item type: {type(item)}, converting to string")
                normalized_evidence.append({
                    "type": "agent_result",
                    "source": "unspecified",
                    "relevance": str(item)
                })
        return normalized_evidence
    else:
        # Unknown type → convert to string and wrap in list
        logger.warning(f"[SPEC_GEN] Unknown evidence type: {type(evidence_value)}, converting to string")
        return [{
            "type": "agent_result",
            "source": "unspecified",
            "relevance": str(evidence_value)
        }]


def _normalize_security_status(status_value: any) -> str:
    """
    Normalize security_status to uppercase string.
    
    Handles:
    - null/None → "REVIEW_REQUIRED"
    - lowercase string → uppercase
    - invalid value → "REVIEW_REQUIRED"
    """
    if status_value is None:
        logger.info("[SPEC_GEN] Normalizing security_status from null to 'REVIEW_REQUIRED'")
        return "REVIEW_REQUIRED"
    
    if not isinstance(status_value, str):
        logger.warning(f"[SPEC_GEN] security_status is not a string: {type(status_value)}, converting to 'REVIEW_REQUIRED'")
        return "REVIEW_REQUIRED"
    
    # Convert to uppercase
    normalized = status_value.upper().strip()
    
    # Validate it's a reasonable security status (basic check)
    # Common values: REVIEW_REQUIRED, SECURE, VULNERABLE, PENDING, etc.
    # If it's empty or looks invalid, default to REVIEW_REQUIRED
    if not normalized or len(normalized) > 50:  # Sanity check
        logger.warning(f"[SPEC_GEN] security_status looks invalid: '{status_value}', converting to 'REVIEW_REQUIRED'")
        return "REVIEW_REQUIRED"
    
    if normalized != status_value:
        logger.info(f"[SPEC_GEN] Normalizing security_status from '{status_value}' to '{normalized}'")
    
    return normalized


def _normalize_output_dict(data: dict, model_class: Type[BaseModel]) -> dict:
    """
    Normalize common data format issues before validation.
    
    Fixes:
    - success_metrics: Converts dictionaries {'metric': '...'} to strings
    - RequirementEnrichmentOutput: Converts 'requirements' key to 'enrichments'
    - RequirementEnrichmentOutput: Converts required_by strings to lists
    - RequirementEnrichmentOutput: Normalizes appendix.evidence (string/list[str] → List[Evidence])
    - RequirementEnrichmentOutput: Normalizes external_dependencies.security_status (null/lowercase → uppercase)
    - Guardrail types: Normalizes to lowercase enum values
    - TechnicalDesignOutput: Converts constraint dicts to strings, rate_limiting dicts to strings
    - Request.body: Converts None to empty dict
    """
    from .spec_models import (
        RequirementsOutputCore,
        RequirementEnrichmentOutput,
        TechnicalDesignOutput,
        GuardrailType,
        Specification,
    )
    
    normalized = data.copy()
    
    # Fix RequirementEnrichmentOutput: 'requirements' -> 'enrichments'
    if model_class == RequirementEnrichmentOutput:
        if 'requirements' in normalized and 'enrichments' not in normalized:
            logger.info("[SPEC_GEN] Normalizing RequirementEnrichmentOutput: 'requirements' -> 'enrichments'")
            normalized['enrichments'] = normalized.pop('requirements')
        
        # Normalize enrichments recursively
        if 'enrichments' in normalized and isinstance(normalized['enrichments'], list):
            for enrichment in normalized['enrichments']:
                if not isinstance(enrichment, dict):
                    continue
                
                # Fix external_dependencies[].required_by: convert string to list
                if 'external_dependencies' in enrichment:
                    deps = enrichment['external_dependencies']
                    if isinstance(deps, list):
                        for dep in deps:
                            if isinstance(dep, dict):
                                # Normalize required_by
                                if 'required_by' in dep:
                                    required_by = dep['required_by']
                                    if isinstance(required_by, str):
                                        logger.info(f"[SPEC_GEN] Normalizing required_by from string '{required_by}' to list")
                                        dep['required_by'] = [required_by]
                                    elif not isinstance(required_by, list):
                                        logger.warning(f"[SPEC_GEN] required_by is neither string nor list: {type(required_by)}, converting to list")
                                        dep['required_by'] = [str(required_by)] if required_by else []
                                
                                # Normalize security_status
                                if 'security_status' in dep:
                                    dep['security_status'] = _normalize_security_status(dep['security_status'])
                
                # Fix guardrails[].type: normalize to lowercase enum values
                if 'guardrails' in enrichment:
                    guardrails = enrichment['guardrails']
                    if isinstance(guardrails, list):
                        for guardrail in guardrails:
                            if isinstance(guardrail, dict) and 'type' in guardrail:
                                guardrail_type = guardrail['type']
                                if isinstance(guardrail_type, str):
                                    # Normalize to lowercase
                                    normalized_type = guardrail_type.lower()
                                    # Map common variations to enum values
                                    type_mapping = {
                                        'must': 'must',
                                        'must_not': 'must_not',
                                        'must-not': 'must_not',
                                        'should': 'should',
                                        'should_not': 'should_not',
                                        'should-not': 'should_not',
                                    }
                                    normalized_type = type_mapping.get(normalized_type, normalized_type)
                                    if normalized_type != guardrail_type:
                                        logger.info(f"[SPEC_GEN] Normalizing guardrail type from '{guardrail_type}' to '{normalized_type}'")
                                        guardrail['type'] = normalized_type
                
                # Fix appendix.evidence: normalize string/list[str] to List[Evidence]
                if 'appendix' in enrichment:
                    appendix = enrichment['appendix']
                    if isinstance(appendix, dict) and 'evidence' in appendix:
                        evidence_value = appendix['evidence']
                        # Check if evidence needs normalization (not already List[Evidence])
                        needs_normalization = False
                        if isinstance(evidence_value, str):
                            needs_normalization = True
                        elif isinstance(evidence_value, list):
                            # Check if any items are strings instead of dicts
                            for item in evidence_value:
                                if isinstance(item, str):
                                    needs_normalization = True
                                    break
                        
                        if needs_normalization:
                            logger.info(f"[SPEC_GEN] Normalizing appendix.evidence from {type(evidence_value).__name__} to List[Evidence]")
                            appendix['evidence'] = _normalize_evidence(evidence_value)
    
    # Fix success_metrics if it's a list of dictionaries instead of strings
    if model_class == RequirementsOutputCore and 'success_metrics' in normalized:
        metrics = normalized['success_metrics']
        if isinstance(metrics, list):
            normalized_metrics = []
            for metric in metrics:
                if isinstance(metric, dict):
                    # Extract the metric value from dictionary
                    if 'metric' in metric:
                        normalized_metrics.append(str(metric['metric']))
                    elif 'description' in metric:
                        normalized_metrics.append(str(metric['description']))
                    elif 'text' in metric:
                        normalized_metrics.append(str(metric['text']))
                    else:
                        # Try to get the first string value
                        for key, value in metric.items():
                            if isinstance(value, str):
                                normalized_metrics.append(value)
                                break
                        else:
                            logger.warning(f"Could not extract metric from dict: {metric}")
                            normalized_metrics.append(str(metric))
                elif isinstance(metric, str):
                    normalized_metrics.append(metric)
                else:
                    normalized_metrics.append(str(metric))
            
            if normalized_metrics != metrics:
                logger.info(f"[SPEC_GEN] Normalized success_metrics from {len(metrics)} items (some dicts) to {len(normalized_metrics)} strings")
                normalized['success_metrics'] = normalized_metrics
    
    # Fix TechnicalDesignOutput: normalize constraints and rate_limiting fields
    if model_class == TechnicalDesignOutput:
        # Normalize data_models[].fields[].constraints (dict -> string)
        if 'data_models' in normalized and isinstance(normalized['data_models'], list):
            for data_model in normalized['data_models']:
                if isinstance(data_model, dict) and 'fields' in data_model:
                    for field in data_model.get('fields', []):
                        if isinstance(field, dict) and 'constraints' in field:
                            constraints = field['constraints']
                            if isinstance(constraints, dict):
                                # Convert dict to JSON string
                                field['constraints'] = json.dumps(constraints)
                            elif constraints is None:
                                field['constraints'] = None
        
        # Normalize interfaces[].request.rate_limiting (dict -> string)
        if 'interfaces' in normalized and isinstance(normalized['interfaces'], list):
            for interface in normalized['interfaces']:
                if isinstance(interface, dict) and 'request' in interface:
                    request = interface['request']
                    if isinstance(request, dict):
                        # Fix rate_limiting: dict -> string
                        if 'rate_limiting' in request:
                            rate_limiting = request['rate_limiting']
                            if isinstance(rate_limiting, dict):
                                request['rate_limiting'] = json.dumps(rate_limiting)
                        
                        # Fix body: None -> empty dict
                        if 'body' in request and request['body'] is None:
                            request['body'] = {}
        
        # Normalize external_dependencies[].security_status
        if 'external_dependencies' in normalized and isinstance(normalized['external_dependencies'], list):
            for dep in normalized['external_dependencies']:
                if isinstance(dep, dict) and 'security_status' in dep:
                    dep['security_status'] = _normalize_security_status(dep['security_status'])
    
    # Fix Specification: normalize external_dependencies_summary[].source
    if model_class == Specification:
        if 'external_dependencies_summary' in normalized and isinstance(normalized['external_dependencies_summary'], list):
            for dep in normalized['external_dependencies_summary']:
                if isinstance(dep, dict):
                    # Add default source if missing
                    if 'source' not in dep or not dep['source']:
                        logger.info("[SPEC_GEN] Normalizing external_dependencies_summary: adding default 'source' field")
                        dep['source'] = "research"
                    # Also normalize security_status if present
                    if 'security_status' in dep:
                        dep['security_status'] = _normalize_security_status(dep['security_status'])
    
    return normalized

SPEC_GEN_TASK_PROMPT = """You are a **CONVERSATIONAL SPECIFICATION GENERATION AGENT** that transforms user requests into comprehensive technical specifications through iterative refinement.

## Your Capabilities

You can work in multiple modes:

### 1. **New Specification Generation**
When the user requests a new specification:
- Execute the full 7-step workflow:
  1. Research → 2. Requirements Core → 3. Requirements Enrichment → 
  4. Architecture → 5. Technical Design → 6. Validation → 7. Assembly
- Generate a complete Specification from scratch

### 2. **Iterative Refinement**
When the user wants to refine an existing specification:
- Extract the previous specification from conversation history
- Understand what the user wants to change
- Regenerate only the affected sections
- Maintain consistency with unchanged sections

### 3. **Partial Updates**
When the user requests specific section updates (e.g., "just update the architecture"):
- Extract previous specification from history
- Execute only the workflow steps needed for that section
- Preserve other sections unchanged
- Re-assemble the complete specification

### 4. **Clarifying Questions**
When the user's request is ambiguous:
- Ask specific clarifying questions
- Don't guess - ask for clarification
- Examples: "Which sections should I update?", "What specific changes do you want?"

## Request Classification

Analyze the user's request and conversation history to determine:

1. **Is this a new request?** (No previous spec in history)
   → Execute full 7-step workflow

2. **Is this a refinement?** (User mentions changes to existing spec)
   → Extract previous spec, identify what to change, regenerate affected sections

3. **Is this a partial update?** (User specifies sections: "update architecture", "regenerate requirements")
   → Extract previous spec, execute only needed workflow steps

4. **Is clarification needed?** (Request is ambiguous)
   → Ask specific questions before proceeding

## Conversation History Analysis

**CRITICAL**: Always check conversation history for:
- Previous specifications (look for JSON blocks with Specification structure)
- User's original request context
- Previous refinement requests
- Any Q&A context

**Extracting Previous Specifications:**
- Look for JSON code blocks containing Specification data
- Parse the JSON to get the previous specification object
- Use it as context for refinements

## How Workflow Execution Works

**IMPORTANT**: You are a conversational coordinator. The actual workflow execution happens automatically based on your analysis:

1. **You analyze** the request and conversation history
2. **You classify** the request type (new/refinement/partial/clarification)
3. **The system executes** the appropriate workflow steps automatically
4. **You present** the results conversationally

## Response Format

**For new specifications:**
- Acknowledge the request
- Explain that you're generating a complete specification
- The system will execute the full 7-step workflow
- Present the resulting Specification JSON when ready

**For refinements:**
- Acknowledge what the user wants to change
- Explain which sections you'll update
- The system will regenerate affected sections
- Present the updated Specification JSON

**For partial updates:**
- Confirm which sections you're updating
- The system will execute only needed workflow steps
- Present the updated Specification JSON

**For clarification:**
- Ask specific, focused questions
- Examples:
  - "Which sections would you like me to update: requirements, architecture, or technical design?"
  - "What specific changes do you want in the architecture section?"
  - "Should I regenerate the entire specification or just specific parts?"
- Wait for user response before proceeding

## Conversation Guidelines

- **Be conversational and helpful** - explain what you're doing
- **Check history first** - always look for previous specifications
- **Ask when unclear** - don't guess what the user wants
- **Explain your process** - tell the user what workflow steps will run
- **Highlight changes** - when refining, explain what's different

## Clarification Phase Handling

**CRITICAL**: For new specifications, clarification answers are provided before generation.

When clarification answers are present in the context:
- **Treat them as authoritative constraints** - user answers override default assumptions
- **Reference clarification decisions in ADRs** - explicitly state architectural choices based on user answers
- **Reflect clarified stack decisions** - ensure technical design aligns with user's clarified preferences
- **Respect integration points** - if user specified integration approach, follow it exactly

Example: If user answered "Use existing user table schema", ensure ADRs and technical design reference this constraint.

## Important Notes

- **Always check conversation history** before executing workflow
- **Preserve unchanged sections** when doing partial updates
- **Ask clarifying questions** when request is ambiguous
- **Maintain conversation context** across multiple turns
- **Use validation loopback** if quality issues detected (up to 3 iterations)
- **Be conversational** - explain what you're doing and why
- **Respect clarification answers** - treat user clarification responses as design constraints
"""


class SpecGenAgent(ChatAgent):
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
        """Build the conversational wrapper agent that orchestrates the spec generation workflow."""
        agent_config = AgentConfig(
            role="Conversational Specification Generation Agent",
            goal="Generate and iteratively refine comprehensive technical specifications through conversation",
            backstory="""
                You are an expert conversational specification generation agent that transforms user requirements
                into detailed technical specifications through a systematic 7-step process. You support iterative
                refinement, allowing users to update specific sections without regenerating the entire specification.
                You maintain conversation context and can extract previous specifications from history to build upon them.
            """,
            tasks=[
                TaskConfig(
                    description=SPEC_GEN_TASK_PROMPT,
                    expected_output="Complete Specification JSON with all 9 mandatory sections, or clarifying questions if request is ambiguous",
                )
            ],
        )
        # Add tools: codebase exploration + conversation state management
        tools = self.tools_provider.get_tools([
            # Codebase exploration tools (same as code_gen_agent)
            "get_code_from_multiple_node_ids",
            "get_node_neighbours_from_node_id",
            "get_code_from_probable_node_name",
            "ask_knowledge_graph_queries",
            "get_nodes_from_tags",
            "get_code_file_structure",
            "webpage_extractor",
            "web_search_tool",
            "fetch_file",
            "analyze_code_structure",
            # Conversation state management tools
            "create_todo",
            "update_todo_status",
            "get_todo",
            "list_todos",
            "add_requirements",
            "get_requirements",
        ])
        return PydanticRagAgent(self.llm_provider, agent_config, tools)

    async def _generate_specification(
        self,
        user_prompt: str,
        qa_context: Optional[str],
        project_id: str,
        project_name: str = "",
        node_ids: Optional[List[str]] = None,
        max_validation_iterations: int = 3,
    ) -> Specification:
        """
        Execute the 7-step spec generation workflow.
        
        Args:
            user_prompt: User's original request
            qa_context: Optional Q&A context string (may include pre-loaded code from node_ids)
            project_id: Project identifier
            project_name: Project name in owner/repo format (for richer context)
            node_ids: Optional list of node IDs that were referenced by the user
            max_validation_iterations: Maximum validation loopback iterations
        
        Returns:
            Complete Specification object
        """
        # Build QA context string
        qa_context_str = qa_context or ""
        full_context = f"USER REQUEST:\n{user_prompt}\n\n"
        if qa_context_str:
            # Check if clarification answers are present
            if "CLARIFICATION ANSWERS" in qa_context_str:
                full_context += f"⚠️ CLARIFICATION ANSWERS PROVIDED - THESE ARE AUTHORITATIVE CONSTRAINTS:\n{qa_context_str}\n\n"
                full_context += "IMPORTANT: All architectural decisions, technical design, and implementation choices must respect and reference these clarification answers.\n\n"
            else:
                full_context += f"Q&A CONTEXT:\n{qa_context_str}\n"
        
        logger.info("[SPEC_GEN] Starting specification generation workflow")
        
        # Step 1: Research
        logger.info("[SPEC_GEN] Step 1/7: Research")
        try:
            research_agent = create_research_agent(self.llm_provider, self.tools_provider)
            from app.modules.intelligence.agents.chat_agent import ChatContext
            
            # Build rich additional context for the research agent
            research_additional_context = f"Project ID: {project_id}"
            if project_name:
                research_additional_context += f"\nProject Name (owner/repo): {project_name}"
            if qa_context:
                research_additional_context += f"\n\nUser context and referenced code:\n{qa_context}"
            
            # Build enhanced query that explicitly mentions pre-loaded code and node_ids
            node_ids_note = ""
            if node_ids and len(node_ids) > 0:
                node_ids_note = f"\n\n⚠️ IMPORTANT: The user has referenced specific code nodes (node_ids: {', '.join(node_ids)}). "
                node_ids_note += "These nodes contain code directly relevant to the specification request. "
                node_ids_note += "If you see pre-loaded code in the context above, explore those areas deeply using "
                node_ids_note += f"`get_code_from_multiple_node_ids` or `get_node_neighbours_from_node_id` to understand "
                node_ids_note += "how they connect to the broader codebase. This code is critical for generating accurate specifications."
            
            research_query = f"""Research this codebase to support specification generation.

{full_context}
{node_ids_note}

## Your Research Task:

1. **Start with repository structure**: Call `get_code_file_structure` with project_id="{project_id}" and path="" to see the full project layout.

2. **Explore the tech stack**: Use `ask_knowledge_graph_queries` with queries like:
   - "What framework/technology stack is used in this project?"
   - "What database or ORM is used?"
   - "What architectural patterns exist?"
   - "What are the main entry points and key modules?"

3. **Read key files**: Use `fetch_file` to read:
   - README.md, package.json/requirements.txt/pyproject.toml
   - Main entry point files (server.js, main.py, app.py, index.ts, etc.)
   - Configuration files

4. **Deep dive into referenced code** (if node_ids were provided):
   - Use `get_code_from_multiple_node_ids` with node_ids={node_ids if node_ids else "[]"} to get the specific code the user referenced
   - Use `get_node_neighbours_from_node_id` to explore how that code connects to other parts of the codebase
   - Use `analyze_code_structure` to understand the structure of files containing referenced code

5. **Understand relationships**: Trace import/dependency relationships to see how components connect.

6. **External research**: Use `web_search_tool` for best practices relevant to the discovered tech stack.

The project_id for all tool calls is: {project_id}"""
            
            research_ctx = ChatContext(
                project_id=project_id,
                project_name=project_name,
                curr_agent_id="research_agent",
                history=[],
                node_ids=node_ids,  # Pass node_ids so research agent can explore them
                query=research_query,
                user_id="system",
                conversation_id="spec-gen-research",
                additional_context=research_additional_context,
            )
            research_result = await research_agent.run(research_ctx)
            
            # Extract ResearchFindings from text response
            research_output: ResearchFindings = self._parse_research_from_text(research_result.response)
            
            logger.info(f"[SPEC_GEN] Research completed with {len(research_output.sources)} sources")
        except Exception as e:
            # Research failure must not block spec generation
            logger.warning(f"[SPEC_GEN] Research step failed: {e}. Continuing with empty research findings.")
            logger.exception("Research step exception details")
            # Create safe fallback ResearchFindings
            # Note: ResearchFindings requires min_length=5 for sources, so we provide 5 placeholder sources
            from .spec_models import ResearchSource
            research_output = ResearchFindings(
                sources=[
                    ResearchSource(
                        query="research_fallback",
                        findings="Research step encountered errors. Specification generation will proceed without research findings.",
                        source="system"
                    ),
                    ResearchSource(
                        query="research_fallback",
                        findings="Knowledge graph or web search tools may be unavailable in test environment.",
                        source="system"
                    ),
                    ResearchSource(
                        query="research_fallback",
                        findings="",
                        source="system"
                    ),
                    ResearchSource(
                        query="research_fallback",
                        findings="",
                        source="system"
                    ),
                    ResearchSource(
                        query="research_fallback",
                        findings="",
                        source="system"
                    ),
                ],
                summary="Research step encountered errors (likely due to missing project or API keys in test environment). Specification generation will proceed without research findings."
            )
            logger.info("[SPEC_GEN] Using degraded research mode - empty research findings")
        
        # Step 2a: Requirements Core
        logger.info("[SPEC_GEN] Step 2a/7: Requirements Core")
        try:
            requirements_agent = create_requirements_agent(self.llm_provider)
            requirements_prompt = self._build_requirements_prompt(full_context, research_output)
            requirements_result = await requirements_agent.run(requirements_prompt)
            requirements_core: RequirementsOutputCore = parse_agent_output(requirements_result, RequirementsOutputCore)
            logger.info(f"[SPEC_GEN] Requirements core: {len(requirements_core.functional_requirements)} FRs, {len(requirements_core.non_functional_requirements)} NFRs, {len(requirements_core.success_metrics)} success metrics")
        except Exception as e:
            error_msg = str(e)
            logger.error(f"[SPEC_GEN] Requirements core step failed: {error_msg}")
            # Check if it's a validation error we can help with
            if "success_metrics" in error_msg.lower() or "validation" in error_msg.lower():
                logger.warning("[SPEC_GEN] Validation error in requirements - this may be due to format issues. The normalization function should have handled this.")
                logger.debug(f"[SPEC_GEN] Error details: {e}")
            raise
        
        # Step 2b: Requirements Enrichment
        logger.info("[SPEC_GEN] Step 2b/7: Requirements Enrichment")
        enrichment_agent = create_enrichment_agent(self.llm_provider)
        enrichment_prompt = self._build_enrichment_prompt(requirements_core, full_context, research_output)
        enrichment_result = await enrichment_agent.run(enrichment_prompt)
        enrichment_output: RequirementEnrichmentOutput = parse_agent_output(enrichment_result, RequirementEnrichmentOutput)
        logger.info(f"[SPEC_GEN] Enrichment completed: {len(enrichment_output.enrichments)} requirements enriched")
        
        # Merge core + enrichments
        requirements_output = self._merge_core_and_enrichments(requirements_core, enrichment_output)
        logger.info(f"[SPEC_GEN] Requirements merged: {len(requirements_output.functional_requirements)} FRs, {len(requirements_output.non_functional_requirements)} NFRs")
        
        # Step 3: Architecture
        logger.info("[SPEC_GEN] Step 3/7: Architecture")
        architecture_agent = create_architecture_agent(self.llm_provider)
        architecture_prompt = self._build_architecture_prompt(full_context, research_output, requirements_output)
        architecture_result = await architecture_agent.run(architecture_prompt)
        architecture_output: ArchitectureOutput = parse_agent_output(architecture_result, ArchitectureOutput)
        logger.info(f"[SPEC_GEN] Architecture completed with {len(architecture_output.adrs)} ADRs")
        
        # Step 4: Technical Design
        logger.info("[SPEC_GEN] Step 4/7: Technical Design")
        technical_agent = create_technical_design_agent(self.llm_provider)
        technical_prompt = self._build_technical_design_prompt(full_context, research_output, requirements_output, architecture_output)
        technical_result = await technical_agent.run(technical_prompt)
        technical_output: TechnicalDesignOutput = parse_agent_output(technical_result, TechnicalDesignOutput)
        logger.info(f"[SPEC_GEN] Technical design: {len(technical_output.data_models)} models, {len(technical_output.interfaces)} interfaces")
        
        # Step 5: Validation with loopback
        logger.info("[SPEC_GEN] Step 5/7: Validation")
        validation_report: Optional[ValidationReport] = None
        
        for iteration in range(max_validation_iterations):
            logger.info(f"[SPEC_GEN] Validation iteration {iteration + 1}/{max_validation_iterations}")
            
            validator_agent = create_validator_agent(self.llm_provider)
            validation_prompt = self._build_validation_prompt(
                requirements_output,
                architecture_output,
                technical_output,
                iteration + 1,
            )
            validation_result = await validator_agent.run(validation_prompt)
            validation_report = parse_agent_output(validation_result, ValidationReport)
            
            if validation_report.passed:
                logger.info(f"[SPEC_GEN] Validation passed on iteration {iteration + 1}")
                break
            
            logger.info(f"[SPEC_GEN] Validation failed: {validation_report.feedback[:200]}...")
            
            if iteration == max_validation_iterations - 1:
                logger.warning(f"[SPEC_GEN] Validation did not pass after {max_validation_iterations} iterations. Proceeding with assembly.")
                break
            
            # Loopback to target step
            target_step = validation_report.target_step or "requirements"
            logger.info(f"[SPEC_GEN] Loopback to step: {target_step}")
            
            if target_step == "requirements":
                # Re-run two-phase requirements with feedback
                requirements_prompt_with_feedback = f"{requirements_prompt}\n\nVALIDATION FEEDBACK (must address):\n{validation_report.feedback}"
                requirements_result = await requirements_agent.run(requirements_prompt_with_feedback)
                requirements_core = parse_agent_output(requirements_result, RequirementsOutputCore)
                enrichment_result = await enrichment_agent.run(
                    self._build_enrichment_prompt(requirements_core, full_context, research_output),
                )
                enrichment_output = parse_agent_output(enrichment_result, RequirementEnrichmentOutput)
                requirements_output = self._merge_core_and_enrichments(requirements_core, enrichment_output)
            
            elif target_step == "architecture":
                architecture_prompt_with_feedback = f"{architecture_prompt}\n\nVALIDATION FEEDBACK (must address):\n{validation_report.feedback}"
                architecture_result = await architecture_agent.run(architecture_prompt_with_feedback)
                architecture_output = parse_agent_output(architecture_result, ArchitectureOutput)
            
            elif target_step == "technical_design":
                technical_prompt_with_feedback = f"{technical_prompt}\n\nVALIDATION FEEDBACK (must address):\n{validation_report.feedback}"
                technical_result = await technical_agent.run(technical_prompt_with_feedback)
                technical_output = parse_agent_output(technical_result, TechnicalDesignOutput)
        
        # Step 6: Assembly
        logger.info("[SPEC_GEN] Step 6/7: Assembly")
        assembler_agent = create_assembler_agent(self.llm_provider)
        assembly_prompt = self._build_assembly_prompt(
            full_context,
            research_output,
            requirements_output,
            architecture_output,
            technical_output,
        )
        assembly_result = await assembler_agent.run(assembly_prompt)
        specification: Specification = parse_agent_output(assembly_result, Specification)
        
        # Validate specification has required fields
        if not specification.tl_dr or not specification.tl_dr.strip():
            raise ValueError("Specification missing required field: tl_dr")
        if not specification.functional_requirements:
            raise ValueError("Specification missing required field: functional_requirements (must have at least 1)")
        if not specification.non_functional_requirements:
            raise ValueError("Specification missing required field: non_functional_requirements (must have at least 1)")
        if not specification.architectural_decisions:
            raise ValueError("Specification missing required field: architectural_decisions (must have at least 1)")
        if not specification.success_metrics or len(specification.success_metrics) < 3:
            raise ValueError(f"Specification missing required field: success_metrics (must have at least 3, got {len(specification.success_metrics) if specification.success_metrics else 0})")
        
        logger.info(f"[SPEC_GEN] Specification assembled successfully with {len(specification.functional_requirements)} FRs, {len(specification.non_functional_requirements)} NFRs, {len(specification.architectural_decisions)} ADRs")
        return specification
    
    def _format_research_context(self, research: ResearchFindings, max_summary_len: int = 4000) -> str:
        """Format research findings into a comprehensive context string for downstream agents.
        
        Includes both the summary AND key source findings so downstream agents
        get file paths, tech stack details, and specific codebase evidence.
        Preserves all critical codebase information without excessive truncation.
        """
        parts = []
        
        # Include full summary (this contains structured sections like tech stack, architecture, etc.)
        # Use generous limit to preserve all structured sections
        summary = research.summary[:max_summary_len]
        parts.append(f"RESEARCH SUMMARY:\n{summary}")
        
        # Include ALL codebase source findings — these have specific file paths, code patterns, etc.
        # Don't truncate too aggressively - these are critical for accurate specifications
        codebase_sources = [s for s in research.sources if s.source in ("explore_agent", "codebase") and s.findings and len(s.findings) > 20]
        if codebase_sources:
            source_details = []
            # Include more sources (up to 10) and allow longer findings (up to 600 chars)
            for s in codebase_sources[:10]:
                findings_preview = s.findings[:600] if len(s.findings) > 600 else s.findings
                source_details.append(f"- [{s.query}]: {findings_preview}")
            parts.append(f"\nKEY CODEBASE FINDINGS ({len(codebase_sources)} sources):\n" + "\n".join(source_details))
        
        # Include web/external research sources
        web_sources = [s for s in research.sources if s.source in ("librarian_agent", "web") and s.findings and len(s.findings) > 20]
        if web_sources:
            web_details = []
            for s in web_sources[:5]:  # Top 5 web sources
                findings_preview = s.findings[:400] if len(s.findings) > 400 else s.findings
                web_details.append(f"- [{s.query}]: {findings_preview}")
            parts.append(f"\nEXTERNAL RESEARCH ({len(web_sources)} sources):\n" + "\n".join(web_details))
        
        # Include any other sources that might have valuable information
        other_sources = [s for s in research.sources if s.source not in ("explore_agent", "codebase", "librarian_agent", "web") and s.findings and len(s.findings) > 20]
        if other_sources:
            other_details = []
            for s in other_sources[:5]:
                findings_preview = s.findings[:300] if len(s.findings) > 300 else s.findings
                other_details.append(f"- [{s.query}]: {findings_preview}")
            parts.append(f"\nADDITIONAL RESEARCH ({len(other_sources)} sources):\n" + "\n".join(other_details))
        
        return "\n\n".join(parts)
    
    def _build_requirements_prompt(self, qa_context: str, research: ResearchFindings) -> str:
        """Build prompt for requirements generation."""
        research_context = self._format_research_context(research, max_summary_len=4000)
        
        clarification_note = ""
        if "CLARIFICATION ANSWERS" in qa_context:
            clarification_note = "\n\n⚠️ CLARIFICATION ANSWERS PROVIDED: User has provided specific answers to architectural questions. These answers are authoritative constraints that MUST be reflected in the requirements. Reference user's clarified preferences in requirement descriptions and acceptance criteria."
        
        return f"""{qa_context}

{research_context}
{clarification_note}

Generate functional requirements (3-5), non-functional requirements (2-3), and success metrics (3-4) based on the above context.
- Requirements MUST reference the actual codebase structure, tech stack, and patterns discovered in the research.
- Use specific file paths and technology names from the research findings.
- If clarification answers were provided, ensure requirements align with user's clarified architectural preferences."""
    
    def _build_enrichment_prompt(self, core_output: RequirementsOutputCore, qa_context: str, research: Optional[ResearchFindings] = None) -> str:
        """Build prompt for requirements enrichment."""
        req_summaries = []
        for fr in core_output.functional_requirements:
            req_summaries.append(f"- {fr.id} ({fr.title}): {fr.description[:200]}")
        for nfr in core_output.non_functional_requirements:
            req_summaries.append(f"- {nfr.id} ({nfr.title}): {nfr.description[:200]}")
        
        research_section = ""
        if research:
            research_section = f"""
CODEBASE CONTEXT (from research):
{research.summary[:2000]}

Use these codebase findings to make enrichments specific to this project — reference actual file paths,
frameworks, and patterns discovered in the research when suggesting implementation recommendations and file impact.
"""
        
        return f"""PROJECT CONTEXT:
{qa_context[:1500]}
{research_section}
REQUIREMENTS TO ENRICH:
{chr(10).join(req_summaries)}

For each requirement ID above, generate enrichment data: guardrails (1 per req),
implementation_recommendations (1-2 per req), external_dependencies (0-2 per req),
file_impact (1-3 per req), and appendix (edge_cases + notes).

Return one RequirementEnrichment per requirement ID."""
    
    def _merge_core_and_enrichments(
        self,
        core_output: RequirementsOutputCore,
        enrichment_output: RequirementEnrichmentOutput,
    ) -> RequirementsOutput:
        """Merge Phase 1 core requirements with Phase 2 enrichments."""
        enrichment_map = {e.requirement_id: e for e in enrichment_output.enrichments}
        
        full_frs = []
        for fr_core in core_output.functional_requirements:
            enrichment = enrichment_map.get(fr_core.id)
            full_frs.append(
                FunctionalRequirement(
                    id=fr_core.id,
                    title=fr_core.title,
                    description=fr_core.description,
                    acceptance_criteria=fr_core.acceptance_criteria,
                    priority=fr_core.priority,
                    dependencies=fr_core.dependencies,
                    guardrails=enrichment.guardrails if enrichment else [],
                    implementation_recommendations=enrichment.implementation_recommendations if enrichment else [],
                    external_dependencies=enrichment.external_dependencies if enrichment else [],
                    file_impact=enrichment.file_impact if enrichment else [],
                    appendix=enrichment.appendix if enrichment else RequirementAppendix(),
                )
            )
        
        full_nfrs = []
        for nfr_core in core_output.non_functional_requirements:
            enrichment = enrichment_map.get(nfr_core.id)
            full_nfrs.append(
                NonFunctionalRequirement(
                    id=nfr_core.id,
                    title=nfr_core.title,
                    category=nfr_core.category,
                    description=nfr_core.description,
                    acceptance_criteria=nfr_core.acceptance_criteria,
                    priority=nfr_core.priority,
                    dependencies=nfr_core.dependencies,
                    measurement_methodology=nfr_core.measurement_methodology,
                    guardrails=enrichment.guardrails if enrichment else [],
                    implementation_recommendations=enrichment.implementation_recommendations if enrichment else [],
                    external_dependencies=enrichment.external_dependencies if enrichment else [],
                    file_impact=enrichment.file_impact if enrichment else [],
                    appendix=enrichment.appendix if enrichment else RequirementAppendix(),
                )
            )
        
        return RequirementsOutput(
            functional_requirements=full_frs,
            non_functional_requirements=full_nfrs,
            success_metrics=core_output.success_metrics,
        )
    
    def _build_architecture_prompt(
        self,
        qa_context: str,
        research: ResearchFindings,
        requirements: RequirementsOutput,
    ) -> str:
        """Build prompt for architecture generation."""
        research_context = self._format_research_context(research, max_summary_len=4000)
        fr_summaries = [f"- {fr.id}: {fr.title}" for fr in requirements.functional_requirements]
        nfr_summaries = [f"- {nfr.id}: {nfr.title}" for nfr in requirements.non_functional_requirements]
        
        clarification_note = ""
        if "CLARIFICATION ANSWERS" in qa_context:
            clarification_note = "\n\n⚠️ CLARIFICATION ANSWERS PROVIDED: User has provided specific architectural preferences. Each ADR MUST explicitly reference and respect these clarification answers. If user specified integration approach, database schema usage, or service architecture, these decisions must be reflected in the ADRs with clear rationale."
        
        return f"""{qa_context}

{research_context}
{clarification_note}

FUNCTIONAL REQUIREMENTS:
{chr(10).join(fr_summaries)}

NON-FUNCTIONAL REQUIREMENTS:
{chr(10).join(nfr_summaries)}

Generate 3-5 architectural decision records (ADRs) with evidence-backed decisions, alternatives considered, and consequences.
- ADRs MUST reference the actual codebase structure, existing patterns, and tech stack from the research findings.
- If the codebase already uses specific patterns (e.g., service layers, middleware, ORM), ADRs should build on them.
- If clarification answers were provided, ensure ADRs explicitly reference user's clarified architectural choices."""
    
    def _build_technical_design_prompt(
        self,
        qa_context: str,
        research: ResearchFindings,
        requirements: RequirementsOutput,
        architecture: ArchitectureOutput,
    ) -> str:
        """Build prompt for technical design generation."""
        research_context = self._format_research_context(research, max_summary_len=3000)
        adr_summaries = [f"- {adr.id}: {adr.title} — {adr.decision[:100]}" for adr in architecture.adrs]
        fr_summaries = [f"- {fr.id}: {fr.title}" for fr in requirements.functional_requirements]
        nfr_summaries = [f"- {nfr.id}: {nfr.title}" for nfr in requirements.non_functional_requirements]
        
        clarification_note = ""
        if "CLARIFICATION ANSWERS" in qa_context:
            clarification_note = "\n\n⚠️ CLARIFICATION ANSWERS PROVIDED: User has specified technical preferences (database schema usage, API structure, service architecture, etc.). Technical design MUST align with these clarified choices. Data models should reflect user's schema preferences, interfaces should match user's API approach, and dependencies should support user's architectural decisions."
        
        return f"""{qa_context}

{research_context}
{clarification_note}

FUNCTIONAL REQUIREMENTS:
{chr(10).join(fr_summaries)}

NON-FUNCTIONAL REQUIREMENTS:
{chr(10).join(nfr_summaries)}

ARCHITECTURAL DECISIONS:
{chr(10).join(adr_summaries)}

Generate data models (3-5), API interfaces (5-10), and external dependencies (5-15) based on the requirements and architecture.
- Data models MUST align with the database/ORM discovered in the research (e.g., if Prisma is used, model definitions should follow Prisma conventions).
- Interfaces should match existing API patterns from the codebase (REST routes, GraphQL, etc.).
- External dependencies should include packages already used in the project where applicable.
- If clarification answers were provided, ensure technical design reflects user's clarified technical preferences."""
    
    def _build_validation_prompt(
        self,
        requirements: RequirementsOutput,
        architecture: ArchitectureOutput,
        technical: TechnicalDesignOutput,
        iteration: int,
    ) -> str:
        """Build prompt for validation."""
        return f"""VALIDATION ITERATION: {iteration}

REQUIREMENTS SUMMARY:
- Functional: {len(requirements.functional_requirements)} FRs
- Non-functional: {len(requirements.non_functional_requirements)} NFRs
- Success metrics: {len(requirements.success_metrics)}

ARCHITECTURE SUMMARY:
- ADRs: {len(architecture.adrs)}

TECHNICAL DESIGN SUMMARY:
- Data models: {len(technical.data_models)}
- Interfaces: {len(technical.interfaces)}
- External dependencies: {len(technical.external_dependencies)}

FULL FUNCTIONAL REQUIREMENTS:
{json.dumps([fr.model_dump() for fr in requirements.functional_requirements], indent=2, default=str)}

FULL NON-FUNCTIONAL REQUIREMENTS:
{json.dumps([nfr.model_dump() for nfr in requirements.non_functional_requirements], indent=2, default=str)}

SUCCESS METRICS:
{json.dumps(requirements.success_metrics, indent=2, default=str)}

FULL ARCHITECTURE:
{json.dumps([adr.model_dump() for adr in architecture.adrs], indent=2, default=str)}

FULL TECHNICAL DESIGN:
{json.dumps(technical.model_dump(), indent=2, default=str)}

Validate completeness, consistency, clarity, traceability, and validity. Report all issues found."""
    
    def _build_assembly_prompt(
        self,
        qa_context: str,
        research: ResearchFindings,
        requirements: RequirementsOutput,
        architecture: ArchitectureOutput,
        technical: TechnicalDesignOutput,
    ) -> str:
        """Build prompt for assembly."""
        research_context = self._format_research_context(research, max_summary_len=3000)
        
        return f"""Assemble the following validated outputs into a complete Specification:

CONTEXT:
{qa_context[:2000]}

{research_context}

REQUIREMENTS OUTPUT:
{json.dumps(requirements.model_dump(), indent=2, default=str)}

ARCHITECTURE OUTPUT:
{json.dumps(architecture.model_dump(), indent=2, default=str)}

TECHNICAL DESIGN OUTPUT:
{json.dumps(technical.model_dump(), indent=2, default=str)}

Combine all outputs into a complete Specification with all 9 mandatory sections.
- The tl_dr and overview must reference the actual project structure and tech stack from research.
- external_dependencies_summary must include dependencies discovered in the research.
- All sections must be grounded in the codebase context provided above."""
    
    def _format_specification(self, specification: Specification) -> str:
        """Format Specification object into a nicely formatted markdown document."""
        from .spec_models import Specification
        
        lines = []
        
        # Title
        lines.append("# Technical Specification")
        lines.append("")
        
        # TL;DR
        lines.append("## Executive Summary")
        lines.append("")
        lines.append(specification.tl_dr)
        lines.append("")
        
        # Context
        if specification.context:
            lines.append("## Context")
            lines.append("")
            if hasattr(specification.context, 'project_name') and specification.context.project_name:
                lines.append(f"**Project**: {specification.context.project_name}")
            if hasattr(specification.context, 'repository_url') and specification.context.repository_url:
                lines.append(f"**Repository**: {specification.context.repository_url}")
            if hasattr(specification.context, 'overview') and specification.context.overview:
                lines.append("")
                lines.append("### Overview")
                lines.append("")
                lines.append(specification.context.overview)
            if specification.context.original_request:
                lines.append("")
                lines.append("### Original Request")
                lines.append("")
                lines.append(specification.context.original_request)
            if specification.context.research_findings:
                lines.append("")
                lines.append("### Research Findings")
                lines.append("")
                lines.append(specification.context.research_findings)
            if specification.context.qa_answers:
                lines.append("")
                lines.append("### Clarification Answers")
                lines.append("")
                lines.append(specification.context.qa_answers)
            lines.append("")
        
        # Success Metrics
        if specification.success_metrics:
            lines.append("## Success Metrics")
            lines.append("")
            for i, metric in enumerate(specification.success_metrics, 1):
                lines.append(f"{i}. {metric}")
            lines.append("")
        
        # Functional Requirements
        if specification.functional_requirements:
            lines.append("## Functional Requirements")
            lines.append("")
            for fr in specification.functional_requirements:
                lines.append(f"### {fr.id}: {fr.title}")
                lines.append("")
                lines.append(f"**Description**: {fr.description}")
                lines.append("")
                if fr.acceptance_criteria:
                    lines.append("**Acceptance Criteria**:")
                    for ac in fr.acceptance_criteria:
                        lines.append(f"- {ac}")
                    lines.append("")
                lines.append(f"**Priority**: {fr.priority.value.title()}")
                if fr.dependencies:
                    lines.append(f"**Dependencies**: {', '.join(fr.dependencies)}")
                lines.append("")
                
                # Enrichments
                if fr.guardrails:
                    lines.append("**Guardrails**:")
                    for guardrail in fr.guardrails:
                        lines.append(f"- **{guardrail.type.value.replace('_', ' ').title()}**: {guardrail.statement}")
                        if guardrail.rationale:
                            lines.append(f"  - *Rationale*: {guardrail.rationale}")
                    lines.append("")
                
                if fr.implementation_recommendations:
                    lines.append("**Implementation Recommendations**:")
                    for rec in fr.implementation_recommendations:
                        lines.append(f"- **{rec.area}**: {rec.recommendation}")
                        if rec.rationale:
                            lines.append(f"  - *Rationale*: {rec.rationale}")
                    lines.append("")
                
                if fr.file_impact:
                    lines.append("**File Impact**:")
                    for impact in fr.file_impact:
                        lines.append(f"- **{impact.action.value.title()}**: `{impact.path}` - {impact.purpose}")
                    lines.append("")
        
        # Non-Functional Requirements
        if specification.non_functional_requirements:
            lines.append("## Non-Functional Requirements")
            lines.append("")
            for nfr in specification.non_functional_requirements:
                lines.append(f"### {nfr.id}: {nfr.title}")
                lines.append("")
                lines.append(f"**Category**: {nfr.category.title()}")
                lines.append("")
                lines.append(f"**Description**: {nfr.description}")
                lines.append("")
                if nfr.acceptance_criteria:
                    lines.append("**Acceptance Criteria**:")
                    for ac in nfr.acceptance_criteria:
                        lines.append(f"- {ac}")
                    lines.append("")
                lines.append(f"**Measurement Methodology**: {nfr.measurement_methodology}")
                lines.append("")
                lines.append(f"**Priority**: {nfr.priority.value.title()}")
                lines.append("")
        
        # Architectural Decisions
        if specification.architectural_decisions:
            lines.append("## Architectural Decision Records (ADRs)")
            lines.append("")
            for adr in specification.architectural_decisions:
                lines.append(f"### {adr.id}: {adr.title}")
                lines.append("")
                lines.append(f"**Status**: {adr.status.value.title()}")
                lines.append("")
                lines.append(f"**Decision**: {adr.decision}")
                lines.append("")
                if adr.context:
                    lines.append(f"**Context**: {adr.context}")
                    lines.append("")
                if adr.alternatives:
                    lines.append("**Alternatives Considered**:")
                    for alt in adr.alternatives:
                        lines.append(f"- {alt}")
                    lines.append("")
                if adr.consequences:
                    lines.append("**Consequences**:")
                    for cons in adr.consequences:
                        lines.append(f"- {cons}")
                    lines.append("")
        
        # Data Models
        if specification.data_models:
            lines.append("## Data Models")
            lines.append("")
            for model in specification.data_models:
                lines.append(f"### {model.name}")
                lines.append("")
                if model.description:
                    lines.append(f"**Description**: {model.description}")
                    lines.append("")
                if model.fields:
                    lines.append("**Fields**:")
                    lines.append("")
                    lines.append("| Field | Type | Constraints | Description |")
                    lines.append("|-------|------|------------|-------------|")
                    for field in model.fields:
                        constraints = field.constraints or "None"
                        desc = field.description or ""
                        lines.append(f"| {field.name} | {field.type} | {constraints} | {desc} |")
                    lines.append("")
        
        # Interfaces
        if specification.interfaces:
            lines.append("## Interfaces")
            lines.append("")
            for interface in specification.interfaces:
                lines.append(f"### {interface.name}")
                lines.append("")
                if interface.description:
                    lines.append(f"**Description**: {interface.description}")
                    lines.append("")
                if interface.request:
                    lines.append("**Request**:")
                    lines.append(f"- **Method**: {interface.request.method}")
                    lines.append(f"- **Path**: {interface.request.path}")
                    if interface.request.body:
                        lines.append(f"- **Body**: {json.dumps(interface.request.body, indent=2)}")
                    if interface.request.rate_limiting:
                        lines.append(f"- **Rate Limiting**: {interface.request.rate_limiting}")
                    lines.append("")
                if interface.response:
                    lines.append("**Response**:")
                    lines.append(f"- **Status**: {interface.response.status_code}")
                    if interface.response.body:
                        lines.append(f"- **Body**: {json.dumps(interface.response.body, indent=2)}")
                    lines.append("")
        
        # External Dependencies
        if specification.external_dependencies_summary:
            lines.append("## External Dependencies")
            lines.append("")
            lines.append("| Name | Version | License | Purpose | Security Status |")
            lines.append("|------|---------|---------|---------|----------------|")
            for dep in specification.external_dependencies_summary:
                security = dep.security_status or "Review Required"
                lines.append(f"| {dep.name} | {dep.version} | {dep.license} | {dep.purpose} | {security} |")
            lines.append("")
        
        # Footer note
        lines.append("---")
        lines.append("")
        lines.append("*This specification was automatically generated. For the JSON format, please request it explicitly.*")
        lines.append("")
        
        return "\n".join(lines)
    
    def _parse_research_from_text(self, text_response: str) -> ResearchFindings:
        """
        Parse research findings from the research agent's text response.
        
        Extracts structured sections (Repository Overview, Technology Stack, Project Structure,
        Architectural Patterns, Key Code Findings, Research Sources, Research Summary) and
        builds a ResearchFindings object that preserves all codebase context for downstream agents.
        """
        from .spec_models import ResearchSource
        
        try:
            # --- Extract all markdown sections ---
            # Split on ## headers to get named sections
            sections = {}
            current_section = "preamble"
            current_content = []
            
            for line in text_response.split("\n"):
                header_match = re.match(r'^#{1,3}\s+(.+)', line)
                if header_match:
                    # Save previous section
                    if current_content:
                        sections[current_section.lower().strip()] = "\n".join(current_content).strip()
                    current_section = header_match.group(1)
                    current_content = []
                else:
                    current_content.append(line)
            # Save last section
            if current_content:
                sections[current_section.lower().strip()] = "\n".join(current_content).strip()
            
            logger.info(f"[SPEC_GEN] Parsed research sections: {list(sections.keys())}")
            
            # --- Build comprehensive summary from ALL context sections ---
            # Combine the richest sections into the summary so downstream agents get full context
            summary_parts = []
            
            # Priority order: these sections contain the most useful codebase context
            context_section_keys = [
                "repository overview",
                "technology stack",
                "project structure",
                "architectural patterns",
                "key code findings",
                "research summary",
                "summary",
            ]
            
            for key in context_section_keys:
                for section_name, section_content in sections.items():
                    if key in section_name and section_content:
                        summary_parts.append(f"### {section_name.title()}\n{section_content}")
                        break
            
            # If we found structured sections, join them into a comprehensive summary
            if summary_parts:
                summary = "\n\n".join(summary_parts)
            else:
                # Fallback: use the full text response (the agent may not have used exact headers)
                summary = text_response
            
            # Limit summary length but be generous — this is the primary context carrier
            summary = summary[:8000]
            
            # --- Extract sources ---
            sources = []
            
            # Look for the Research Sources section
            sources_content = ""
            for section_name, section_content in sections.items():
                if "source" in section_name.lower():
                    sources_content = section_content
                    break
            
            if sources_content:
                # Parse individual source entries — handle multiple formats:
                # Format 1: **Query**: ... / **Findings**: ... / **Source Type**: ...
                # Format 2: - Query: ... / Findings: ... / Source Type: ...
                # Format 3: Numbered list 1. **Query**: ...
                source_blocks = re.split(r'\n(?=\d+\.\s|\*\s\*\*Query|\-\s\*\*Query)', sources_content)
                
                for block in source_blocks:
                    block = block.strip()
                    if not block or len(block) < 20:
                        continue
                    
                    query_match = re.search(r'\*?\*?Query\*?\*?[:\s]+(.+?)(?:\n|$)', block, re.IGNORECASE)
                    findings_match = re.search(r'\*?\*?Findings?\*?\*?[:\s]+(.+?)(?=\n\s*\*?\*?(?:Source|Ref)|$)', block, re.DOTALL | re.IGNORECASE)
                    source_type_match = re.search(r'\*?\*?Source\s*Type\*?\*?[:\s]+"?(codebase|web|explore|librarian)"?', block, re.IGNORECASE)
                    
                    query = query_match.group(1).strip() if query_match else "Codebase exploration"
                    findings = findings_match.group(1).strip() if findings_match else block[:500]
                    
                    source_type = "explore_agent"
                    if source_type_match:
                        st = source_type_match.group(1).lower()
                        source_type = "librarian_agent" if st in ("web", "librarian") else "explore_agent"
                    
                    sources.append(ResearchSource(
                        query=query[:500],
                        findings=findings[:2000],
                        source=source_type,
                    ))
            
            # If we couldn't extract enough structured sources, create them from section content
            if len(sources) < 5:
                # Use the discovered context sections as sources
                for section_name, section_content in sections.items():
                    if len(sources) >= 10:
                        break
                    if section_name in ("preamble", "research sources", "sources"):
                        continue
                    if section_content and len(section_content) > 30:
                        sources.append(ResearchSource(
                            query=f"Codebase analysis: {section_name}",
                            findings=section_content[:2000],
                            source="explore_agent",
                        ))
            
            # Pad to minimum 5 sources if needed
            while len(sources) < 5:
                sources.append(ResearchSource(
                    query="Codebase research",
                    findings="Additional codebase findings from tool exploration.",
                    source="explore_agent",
                ))
            
            result = ResearchFindings(
                sources=sources[:10],
                summary=summary,
            )
            logger.info(f"[SPEC_GEN] Parsed research: {len(result.sources)} sources, summary length={len(result.summary)}")
            return result
            
        except Exception as e:
            logger.warning(f"[SPEC_GEN] Failed to parse research from text, using full response as summary: {e}")
            from .spec_models import ResearchSource
            # Fallback: use the entire response as the summary — don't lose context
            return ResearchFindings(
                sources=[
                    ResearchSource(
                        query="Full codebase research",
                        findings=text_response[:2000],
                        source="explore_agent"
                    ),
                    ResearchSource(query="Codebase structure", findings="See summary for details.", source="explore_agent"),
                    ResearchSource(query="Technology stack", findings="See summary for details.", source="explore_agent"),
                    ResearchSource(query="Architectural patterns", findings="See summary for details.", source="explore_agent"),
                    ResearchSource(query="Key code findings", findings="See summary for details.", source="explore_agent"),
                ],
                summary=text_response[:8000],
            )
    
    async def _extract_previous_spec(self, history: list[str]) -> Optional[Specification]:
        """
        Deterministically extract the most recent valid Specification from history.
        
        Algorithm:
        1. Iterate through history from newest to oldest
        2. Extract JSON blocks using three strategies: fenced json, generic fenced, raw brace matching
        3. Parse each JSON block
        4. Validate using Specification.model_validate() - no heuristics, no manual reconstruction
        5. Return the first valid Specification found
        
        The first JSON block (from newest to oldest) that successfully validates as a Specification
        is the previous spec. No guessing, no length checks, no manual field inference.
        """
        import re
        
        for message in reversed(history):
            if not message or not message.strip():
                continue
            
            # 1️⃣ Extract fenced JSON blocks first (```json ... ```)
            fenced_blocks = re.findall(r'```json\s*(.*?)```', message, re.DOTALL | re.IGNORECASE)
            
            # 2️⃣ Extract generic fenced blocks (``` ... ```)
            generic_blocks = re.findall(r'```\s*(.*?)```', message, re.DOTALL)
            
            # 3️⃣ Extract raw JSON objects using brace matching
            raw_blocks = []
            brace_stack = []
            start = None
            for i, char in enumerate(message):
                if char == '{':
                    if not brace_stack:
                        start = i
                    brace_stack.append(char)
                elif char == '}':
                    if brace_stack:
                        brace_stack.pop()
                        if not brace_stack and start is not None:
                            raw_blocks.append(message[start:i+1])
                            start = None
            
            # Combine all candidate blocks (fenced json first, then generic, then raw)
            candidate_blocks = fenced_blocks + generic_blocks + raw_blocks
            
            # Process blocks from newest to oldest (reverse order)
            for block in reversed(candidate_blocks):
                block = block.strip()
                if not block or not block.startswith('{'):
                    continue
                
                # Try to parse JSON with basic syntax cleanup
                try:
                    parsed = json.loads(block)
                except json.JSONDecodeError:
                    # Try basic syntax fixes (trailing commas, etc.)
                    try:
                        # Remove trailing commas before closing braces/brackets
                        fixed_json = re.sub(r',\s*}', '}', block)
                        fixed_json = re.sub(r',\s*]', ']', fixed_json)
                        parsed = json.loads(fixed_json)
                    except (json.JSONDecodeError, Exception):
                        # If basic fixes don't work, skip this block
                        continue
                
                # Normalize before validation (using the same normalization as parse_agent_output)
                # This handles cases where history contains specs with evidence/security_status issues
                try:
                    normalized_dict = _normalize_output_dict(parsed, Specification)
                except Exception:
                    # Normalization failed, skip this block
                    continue
                
                # Validate using Specification.model_validate() - this is the only truth source
                try:
                    spec = Specification.model_validate(normalized_dict)
                    logger.info("[SPEC_GEN] Valid previous specification found in history.")
                    return spec
                except Exception:
                    # Not a valid Specification, try next block
                    continue
        
        logger.info("[SPEC_GEN] No previous specification found in history")
        return None
    
    async def _classify_request(self, query: str, history: list[str]) -> tuple[str, Optional[list[str]]]:
        """
        Classify the request type and identify target sections if partial update.
        
        Returns:
            (request_type, target_sections)
            request_type: 'new', 'refinement', 'partial', or 'clarification'
            target_sections: List of section names for partial updates, None otherwise
        """
        query_lower = query.lower()
        
        # Check for partial update keywords
        partial_keywords = {
            'architecture': ['architecture', 'architectural', 'adr', 'adrs'],
            'requirements': ['requirements', 'requirement', 'fr', 'nfr', 'functional', 'non-functional'],
            'technical_design': ['technical design', 'data model', 'interface', 'api', 'endpoint'],
            'research': ['research', 'findings'],
        }
        
        target_sections = []
        for section, keywords in partial_keywords.items():
            if any(kw in query_lower for kw in keywords):
                target_sections.append(section)
        
        # Check if there's a previous spec
        previous_spec = await self._extract_previous_spec(history)
        
        logger.info(f"[SPEC_GEN] Classification - previous_spec: {previous_spec is not None}, target_sections: {target_sections}, query: {query[:100]}")
        
        # Classification logic
        if previous_spec is None:
            logger.info("[SPEC_GEN] Classified as 'new' - no previous spec found")
            return ('new', None)
        
        # Check for clarification keywords first
        clarification_keywords = ['what', 'which', 'how', 'can i', 'what sections', 'what can']
        if any(kw in query_lower for kw in clarification_keywords):
            logger.info("[SPEC_GEN] Classified as 'clarification' - clarification keywords detected")
            return ('clarification', None)
        
        # Check for partial update keywords (must come before refinement check)
        partial_update_keywords = ['just', 'only', 'regenerate', 'recreate', 'redo']
        if any(kw in query_lower for kw in partial_update_keywords) and target_sections:
            logger.info(f"[SPEC_GEN] Classified as 'partial' - partial update keywords with target sections: {target_sections}")
            return ('partial', target_sections)
        
        # Check for refinement keywords
        refinement_keywords = ['update', 'change', 'modify', 'refine', 'improve', 'fix', 'correct', 'adjust']
        if any(kw in query_lower for kw in refinement_keywords):
            if target_sections:
                logger.info(f"[SPEC_GEN] Classified as 'partial' - refinement keywords with target sections: {target_sections}")
                return ('partial', target_sections)
            else:
                logger.info("[SPEC_GEN] Classified as 'refinement' - refinement keywords detected")
                return ('refinement', None)
        
        # If previous spec exists but request is ambiguous
        if previous_spec and not target_sections and not any(kw in query_lower for kw in refinement_keywords):
            logger.info("[SPEC_GEN] Classified as 'clarification' - ambiguous request with previous spec")
            return ('clarification', None)
        
        # Default to refinement if previous spec exists
        logger.info("[SPEC_GEN] Classified as 'refinement' - default for existing previous spec")
        return ('refinement', None)
    
    async def _execute_partial_workflow(
        self,
        target_sections: list[str],
        previous_spec: Specification,
        user_request: str,
        qa_context: Optional[str],
        project_id: str,
        project_name: str = "",
        node_ids: Optional[List[str]] = None,
    ) -> Specification:
        """Execute workflow steps only for specified sections."""
        logger.info(f"[SPEC_GEN] Partial workflow for sections: {target_sections}")
        
        # Determine which workflow steps are needed
        needs_research = 'research' in target_sections
        needs_requirements = 'requirements' in target_sections
        needs_architecture = 'architecture' in target_sections
        needs_technical_design = 'technical_design' in target_sections
        
        # Build context
        full_context = f"USER REQUEST:\n{user_request}\n\n"
        if qa_context:
            full_context += f"Q&A CONTEXT:\n{qa_context}\n"
        
        # Execute needed steps
        research_output = None
        requirements_output = None
        architecture_output = None
        technical_output = None
        
        # Research (if needed or if requirements/architecture need it)
        if needs_research or needs_requirements or needs_architecture:
            logger.info("[SPEC_GEN] Executing research step")
            research_agent = create_research_agent(self.llm_provider, self.tools_provider)
            
            # Build rich additional context for the research agent
            partial_research_additional_context = f"Project ID: {project_id}"
            if project_name:
                partial_research_additional_context += f"\nProject Name (owner/repo): {project_name}"
            if qa_context:
                partial_research_additional_context += f"\n\nUser context and referenced code:\n{qa_context}"
            
            # Build enhanced query that explicitly mentions pre-loaded code and node_ids
            node_ids_note = ""
            if node_ids and len(node_ids) > 0:
                node_ids_note = f"\n\n⚠️ IMPORTANT: The user has referenced specific code nodes (node_ids: {', '.join(node_ids)}). "
                node_ids_note += "These nodes contain code directly relevant to the specification request. "
                node_ids_note += "If you see pre-loaded code in the context above, explore those areas deeply using "
                node_ids_note += f"`get_code_from_multiple_node_ids` or `get_node_neighbours_from_node_id` to understand "
                node_ids_note += "how they connect to the broader codebase. This code is critical for generating accurate specifications."
            
            partial_research_query = f"""Research this codebase to support a partial specification update.

{full_context}
{node_ids_note}

## Your Research Task:

1. **Start with repository structure**: Call `get_code_file_structure` with project_id="{project_id}" and path="" to see the full project layout.

2. **Explore the tech stack**: Use `ask_knowledge_graph_queries` with queries like:
   - "What framework/technology stack is used in this project?"
   - "What database or ORM is used?"
   - "What architectural patterns exist?"
   - "What are the main entry points and key modules?"

3. **Read key files**: Use `fetch_file` to read:
   - README.md, package.json/requirements.txt/pyproject.toml
   - Main entry point files (server.js, main.py, app.py, index.ts, etc.)
   - Configuration files

4. **Deep dive into referenced code** (if node_ids were provided):
   - Use `get_code_from_multiple_node_ids` with node_ids={node_ids if node_ids else "[]"} to get the specific code the user referenced
   - Use `get_node_neighbours_from_node_id` to explore how that code connects to other parts of the codebase
   - Use `analyze_code_structure` to understand the structure of files containing referenced code

5. **Understand relationships**: Trace import/dependency relationships to see how components connect.

6. **External research**: Use `web_search_tool` for best practices relevant to the discovered tech stack.

The project_id for all tool calls is: {project_id}"""
            
            research_ctx = ChatContext(
                project_id=project_id,
                project_name=project_name,
                curr_agent_id="research_agent",
                history=[],
                node_ids=node_ids,  # Pass node_ids so research agent can explore them
                query=partial_research_query,
                user_id="system",
                conversation_id="spec-gen-research",
                additional_context=partial_research_additional_context,
            )
            research_result = await research_agent.run(research_ctx)
            # Parse the text response into structured ResearchFindings
            research_output = self._parse_research_from_text(research_result.response)
            logger.info(f"[SPEC_GEN] Partial workflow research: {len(research_output.sources)} sources, summary length={len(research_output.summary)}")
        
        # Requirements (if needed)
        if needs_requirements:
            logger.info("[SPEC_GEN] Executing requirements steps")
            requirements_agent = create_requirements_agent(self.llm_provider)
            requirements_prompt = self._build_requirements_prompt(full_context, research_output or ResearchFindings(sources=[], summary=""))
            requirements_result = await requirements_agent.run(requirements_prompt)
            requirements_core = parse_agent_output(requirements_result, RequirementsOutputCore)
            
            enrichment_agent = create_enrichment_agent(self.llm_provider)
            enrichment_prompt = self._build_enrichment_prompt(requirements_core, full_context, research_output)
            enrichment_result = await enrichment_agent.run(enrichment_prompt)
            enrichment_output = parse_agent_output(enrichment_result, RequirementEnrichmentOutput)
            requirements_output = self._merge_core_and_enrichments(requirements_core, enrichment_output)
        else:
            # Use previous requirements
            requirements_output = RequirementsOutput(
                functional_requirements=previous_spec.functional_requirements,
                non_functional_requirements=previous_spec.non_functional_requirements,
                success_metrics=previous_spec.success_metrics,
            )
        
        # Architecture (if needed)
        if needs_architecture:
            logger.info("[SPEC_GEN] Executing architecture step")
            architecture_agent = create_architecture_agent(self.llm_provider)
            architecture_prompt = self._build_architecture_prompt(
                full_context,
                research_output or ResearchFindings(sources=[], summary=""),
                requirements_output,
            )
            architecture_result = await architecture_agent.run(architecture_prompt)
            architecture_output = parse_agent_output(architecture_result, ArchitectureOutput)
        else:
            # Use previous architecture
            architecture_output = ArchitectureOutput(adrs=previous_spec.architectural_decisions)
        
        # Technical Design (if needed)
        if needs_technical_design:
            logger.info("[SPEC_GEN] Executing technical design step")
            technical_agent = create_technical_design_agent(self.llm_provider)
            technical_prompt = self._build_technical_design_prompt(
                full_context,
                research_output or ResearchFindings(sources=[], summary=""),
                requirements_output,
                architecture_output,
            )
            technical_result = await technical_agent.run(technical_prompt)
            technical_output = parse_agent_output(technical_result, TechnicalDesignOutput)
        else:
            # Use previous technical design
            technical_output = TechnicalDesignOutput(
                data_models=previous_spec.data_models,
                interfaces=previous_spec.interfaces,
                external_dependencies=previous_spec.external_dependencies_summary,
            )
        
        # Always validate and assemble
        logger.info("[SPEC_GEN] Validating and assembling")
        validator_agent = create_validator_agent(self.llm_provider)
        validation_prompt = self._build_validation_prompt(
            requirements_output,
            architecture_output,
            technical_output,
            1,
        )
        validation_result = await validator_agent.run(validation_prompt)
        
        assembler_agent = create_assembler_agent(self.llm_provider)
        assembly_prompt = self._build_assembly_prompt(
            full_context,
            research_output or ResearchFindings(sources=[], summary=""),
            requirements_output,
            architecture_output,
            technical_output,
        )
        assembly_result = await assembler_agent.run(assembly_prompt)
        specification = parse_agent_output(assembly_result, Specification)
        
        # Validate specification has required fields
        if not specification.tl_dr or not specification.tl_dr.strip():
            raise ValueError("Specification missing required field: tl_dr")
        if not specification.functional_requirements:
            raise ValueError("Specification missing required field: functional_requirements (must have at least 1)")
        if not specification.non_functional_requirements:
            raise ValueError("Specification missing required field: non_functional_requirements (must have at least 1)")
        if not specification.architectural_decisions:
            raise ValueError("Specification missing required field: architectural_decisions (must have at least 1)")
        if not specification.success_metrics or len(specification.success_metrics) < 3:
            raise ValueError(f"Specification missing required field: success_metrics (must have at least 3, got {len(specification.success_metrics) if specification.success_metrics else 0})")
        
        logger.info(f"[SPEC_GEN] Partial specification assembled successfully with {len(specification.functional_requirements)} FRs, {len(specification.non_functional_requirements)} NFRs, {len(specification.architectural_decisions)} ADRs")
        return specification
    
    async def _inspect_repo_context(self, project_id: str) -> dict:
        """
        Inspect repository context to gather information about project structure, tech stack, and patterns.
        
        Returns a dictionary with repo findings including:
        - project_structure: File structure overview
        - framework: Detected framework (Express, Django, Next.js, etc.)
        - package_manager: Detected package manager (npm, pip, etc.)
        - database: Database presence and type
        - api_routes: API route patterns if found
        - auth_implementation: Existing auth patterns if found
        - architectural_patterns: Detected patterns
        """
        repo_context = {
            "project_structure": "",
            "framework": "",
            "package_manager": "",
            "database": "",
            "api_routes": "",
            "auth_implementation": "",
            "architectural_patterns": "",
        }
        
        try:
            # Get file structure
            try:
                file_structure_tool = self.tools_provider.tools.get("get_code_file_structure")
                if file_structure_tool:
                    import asyncio
                    # Use coroutine if available, otherwise use func in thread
                    if hasattr(file_structure_tool, 'coroutine') and file_structure_tool.coroutine:
                        structure_result = await file_structure_tool.coroutine(project_id=project_id, path="")
                    else:
                        structure_result = await asyncio.to_thread(file_structure_tool.func, project_id=project_id, path="")
                    if structure_result:
                        repo_context["project_structure"] = str(structure_result)[:2000]  # Limit length
            except Exception as e:
                logger.debug(f"[SPEC_GEN] Could not fetch file structure: {e}")
            
            # Try to detect framework and tech stack using knowledge graph queries
            try:
                kg_tool = self.tools_provider.tools.get("ask_knowledge_graph_queries")
                if kg_tool:
                    # Query for framework detection
                    framework_queries = [
                        "What web framework is used in this project?",
                        "What package manager configuration files exist?",
                        "What database or ORM is used?",
                    ]
                    
                    import asyncio
                    for query in framework_queries:
                        try:
                            # Use coroutine if available, otherwise use func in thread
                            if hasattr(kg_tool, 'coroutine') and kg_tool.coroutine:
                                result = await kg_tool.coroutine(queries=[query], project_id=project_id, node_ids=[])
                            else:
                                result = await asyncio.to_thread(kg_tool.func, queries=[query], project_id=project_id, node_ids=[])
                            
                            if result and isinstance(result, dict):
                                # Result is a dict with query as key
                                findings = ""
                                for key, value in result.items():
                                    if isinstance(value, str):
                                        findings += value + " "
                                    elif isinstance(value, dict):
                                        findings += str(value.get("findings", "")) + " "
                                
                                if findings:
                                    # Detect framework
                                    findings_lower = findings.lower()
                                    if "express" in findings_lower or "express.js" in findings_lower:
                                        repo_context["framework"] = "Express.js"
                                    elif "django" in findings_lower:
                                        repo_context["framework"] = "Django"
                                    elif "flask" in findings_lower:
                                        repo_context["framework"] = "Flask"
                                    elif "next.js" in findings_lower or "nextjs" in findings_lower:
                                        repo_context["framework"] = "Next.js"
                                    elif "react" in findings_lower:
                                        repo_context["framework"] = "React"
                                    
                                    # Detect package manager
                                    if "package.json" in findings_lower:
                                        repo_context["package_manager"] = "npm/yarn"
                                    elif "requirements.txt" in findings_lower or "pyproject.toml" in findings_lower:
                                        repo_context["package_manager"] = "pip"
                                    
                                    # Detect database
                                    if "prisma" in findings_lower:
                                        repo_context["database"] = "Prisma ORM"
                                    elif "sequelize" in findings_lower:
                                        repo_context["database"] = "Sequelize ORM"
                                    elif "sqlalchemy" in findings_lower:
                                        repo_context["database"] = "SQLAlchemy ORM"
                                    elif "postgresql" in findings_lower or "postgres" in findings_lower:
                                        repo_context["database"] = "PostgreSQL"
                                    elif "mysql" in findings_lower:
                                        repo_context["database"] = "MySQL"
                                    elif "mongodb" in findings_lower or "mongoose" in findings_lower:
                                        repo_context["database"] = "MongoDB"
                        except Exception as e:
                            logger.debug(f"[SPEC_GEN] Knowledge graph query failed: {e}")
                            continue
            except Exception as e:
                logger.debug(f"[SPEC_GEN] Could not query knowledge graph: {e}")
            
            # Try to fetch key config files
            try:
                fetch_file_tool = self.tools_provider.tools.get("fetch_file")
                if fetch_file_tool:
                    config_files = ["package.json", "requirements.txt", "pyproject.toml", "composer.json"]
                    import asyncio
                    for config_file in config_files:
                        try:
                            # Use coroutine if available, otherwise use func in thread
                            if hasattr(fetch_file_tool, 'coroutine') and fetch_file_tool.coroutine:
                                result = await fetch_file_tool.coroutine(project_id=project_id, file_path=config_file)
                            else:
                                result = await asyncio.to_thread(fetch_file_tool.func, project_id=project_id, file_path=config_file)
                            
                            if result:
                                content = str(result)[:500]
                                if "package.json" in config_file:
                                    repo_context["package_manager"] = "npm/yarn"
                                    if "express" in content.lower():
                                        repo_context["framework"] = "Express.js"
                                    elif "next" in content.lower():
                                        repo_context["framework"] = "Next.js"
                                elif "requirements.txt" in config_file or "pyproject.toml" in config_file:
                                    repo_context["package_manager"] = "pip"
                                    if "django" in content.lower():
                                        repo_context["framework"] = "Django"
                                    elif "flask" in content.lower():
                                        repo_context["framework"] = "Flask"
                                break
                        except Exception:
                            continue
            except Exception as e:
                logger.debug(f"[SPEC_GEN] Could not fetch config files: {e}")
            
        except Exception as e:
            logger.warning(f"[SPEC_GEN] Error inspecting repo context: {e}")
        
        return repo_context
    
    async def _check_clarification_completed(self, history: list[str]) -> bool:
        """
        Check if clarification questions have been asked and answered.
        
        Returns True if:
        - Previous assistant message contains numbered clarification questions (1., 2., 3., etc.)
        - AND there's a user message after those questions
        - AND the user message is not just another question
        """
        if not history or len(history) < 2:
            return False
        
        # Look for clarification questions in the last assistant message
        last_assistant_msg = None
        for msg in reversed(history):
            if msg.startswith("AI:") or "Before generating" in msg or "clarification" in msg.lower():
                last_assistant_msg = msg
                break
        
        if not last_assistant_msg:
            return False
        
        # Check if it contains numbered MCQ questions (1., 2., 3., etc. with A., B., C. options)
        import re
        # Look for MCQ format: numbered questions with options
        mcq_pattern = r'\n\d+\.\s+[^\n]+(?:\n\s+[A-D]\.\s+[^\n]+)+'
        mcq_questions = re.findall(mcq_pattern, last_assistant_msg, re.MULTILINE)
        
        # Also check for simple numbered questions as fallback
        question_pattern = r'\n\d+\.\s+[^\n]+'
        questions = re.findall(question_pattern, last_assistant_msg)
        
        # Need at least 3 questions (prefer MCQ format)
        if len(mcq_questions) < 3 and len(questions) < 3:
            return False
        
        # Check if there's a user response after the questions
        # The user response should be after the assistant message with questions
        assistant_idx = None
        for i, msg in enumerate(history):
            if msg == last_assistant_msg or (msg.startswith("AI:") and questions[0] in msg):
                assistant_idx = i
                break
        
        if assistant_idx is None or assistant_idx >= len(history) - 1:
            return False
        
        # Check if next message is from user
        next_msg = history[assistant_idx + 1] if assistant_idx + 1 < len(history) else None
        if next_msg and (next_msg.startswith("User:") or not next_msg.startswith("AI:")):
            # User has responded, clarification is complete
            return True
        
        return False
    
    async def _clarification_phase(self, ctx: ChatContext) -> ChatAgentResponse:
        """
        Mandatory clarification phase for new specifications.
        
        Uses the wrapper agent (which has codebase tools) to explore the repo
        and generate repo-aware MCQ clarifying questions.
        """
        logger.info("[SPEC_GEN] Entering clarification phase for new specification")
        
        # Let the wrapper agent explore the codebase and generate MCQ questions.
        # The agent has all codebase tools (get_code_file_structure, fetch_file, etc.)
        # so it can discover the tech stack, patterns, and structure — then ask
        # relevant questions. This is the same approach code_gen_agent uses.
        clarification_prompt = f"""You are preparing to generate a technical specification. Before you do, you need to understand this codebase and ask the user 3-5 clarifying questions.

## Step 1: Explore the codebase (MANDATORY)

You MUST call these tools first to understand the repository:

1. Call `get_code_file_structure` with project_id="{ctx.project_id}" and path="" to see the project layout.
2. Call `fetch_file` to read key files like README.md, package.json, requirements.txt, or the main entry point.
3. Call `ask_knowledge_graph_queries` to understand the framework, database, and architecture.

## Step 2: Generate MCQ questions

Based on what you discovered from the tools AND the user's request below, generate exactly 3-5 multiple-choice questions.

USER REQUEST: {ctx.query}

Each question must:
- Reference specific things you found in the codebase (framework name, file paths, existing patterns)
- Have 3-4 options labeled A, B, C, D
- Help determine architectural decisions for the specification
- Be numbered (1., 2., 3., etc.)

## Output Format

Your ENTIRE response must follow this exact format:

Before generating the specification, I need clarification on a few points based on your request and the current repository structure.

Please select one option (A, B, C, or D) for each question:

1. [Question referencing codebase findings]
   A. [Option]
   B. [Option]
   C. [Option]
   D. [Option]

2. [Question referencing codebase findings]
   A. [Option]
   B. [Option]
   C. [Option]

... (3-5 questions total)

**How to respond:** Simply list your choices, for example:
1. A
2. B
3. C

Once I have your answers, I will generate a detailed and aligned technical specification.

## IMPORTANT
- You MUST call tools first before generating questions.
- Do NOT ask generic questions. Every question must reference what you found in the codebase.
- Do NOT output anything other than the clarification questions in the format above.
- Do NOT generate a specification yet."""
        
        try:
            agent = self._build_agent()
            temp_ctx = ChatContext(
                project_id=ctx.project_id,
                project_name=ctx.project_name,
                curr_agent_id=ctx.curr_agent_id,
                history=[],
                query=clarification_prompt,
                user_id=ctx.user_id,
                conversation_id=ctx.conversation_id,
                additional_context=f"Project ID: {ctx.project_id}",
            )
            response = await agent.run(temp_ctx)
            
            return ChatAgentResponse(
                response=response.response,
                tool_calls=[],
                citations=[],
            )
            
        except Exception as e:
            logger.error(f"[SPEC_GEN] Error in clarification phase: {e}")
            logger.exception("Clarification phase exception details")
            # Fallback to basic MCQ questions
            questions_formatted = self._generate_fallback_mcq_questions(ctx.query, {})
            
            return ChatAgentResponse(
                response=f"""Before generating the specification, I need clarification on a few points.

Please select one option (A, B, C, or D) for each question:

{questions_formatted}

**How to respond:** Simply list your choices, for example:
1. A
2. B
3. C
4. A

Once I have your answers, I will generate a detailed technical specification.""",
                tool_calls=[],
                citations=[],
            )
    
    async def _clarification_phase_stream(self, ctx: ChatContext) -> AsyncGenerator[ChatAgentResponse, None]:
        """
        Streaming version of the clarification phase.
        
        Lets the user see tool calls (file structure, file reads, etc.) happening
        in real time as the agent explores the codebase before generating MCQs.
        """
        logger.info("[SPEC_GEN] Entering clarification phase (streaming)")
        
        clarification_prompt = f"""You are preparing to generate a technical specification. Before you do, you need to understand this codebase and ask the user 3-5 clarifying questions.

## Step 1: Explore the codebase (MANDATORY)

You MUST call these tools first to understand the repository:

1. Call `get_code_file_structure` with project_id="{ctx.project_id}" and path="" to see the project layout.
2. Call `fetch_file` to read key files like README.md, package.json, requirements.txt, or the main entry point.
3. Call `ask_knowledge_graph_queries` to understand the framework, database, and architecture.

## Step 2: Generate MCQ questions

Based on what you discovered from the tools AND the user's request below, generate exactly 3-5 multiple-choice questions.

USER REQUEST: {ctx.query}

Each question must:
- Reference specific things you found in the codebase (framework name, file paths, existing patterns)
- Have 3-4 options labeled A, B, C, D
- Help determine architectural decisions for the specification
- Be numbered (1., 2., 3., etc.)

## Output Format

Your ENTIRE response must follow this exact format:

Before generating the specification, I need clarification on a few points based on your request and the current repository structure.

Please select one option (A, B, C, or D) for each question:

1. [Question referencing codebase findings]
   A. [Option]
   B. [Option]
   C. [Option]
   D. [Option]

2. [Question referencing codebase findings]
   A. [Option]
   B. [Option]
   C. [Option]

... (3-5 questions total)

**How to respond:** Simply list your choices, for example:
1. A
2. B
3. C

Once I have your answers, I will generate a detailed and aligned technical specification.

## IMPORTANT
- You MUST call tools first before generating questions.
- Do NOT ask generic questions. Every question must reference what you found in the codebase.
- Do NOT output anything other than the clarification questions in the format above.
- Do NOT generate a specification yet."""
        
        try:
            agent = self._build_agent()
            temp_ctx = ChatContext(
                project_id=ctx.project_id,
                project_name=ctx.project_name,
                curr_agent_id=ctx.curr_agent_id,
                history=[],
                query=clarification_prompt,
                user_id=ctx.user_id,
                conversation_id=ctx.conversation_id,
                additional_context=f"Project ID: {ctx.project_id}",
            )
            async for chunk in agent.run_stream(temp_ctx):
                yield chunk
        except Exception as e:
            logger.error(f"[SPEC_GEN] Error in clarification phase stream: {e}")
            logger.exception("Clarification phase stream exception details")
            questions_formatted = self._generate_fallback_mcq_questions(ctx.query, {})
            yield ChatAgentResponse(
                response=f"""Before generating the specification, I need clarification on a few points.

Please select one option (A, B, C, or D) for each question:

{questions_formatted}

**How to respond:** Simply list your choices, for example:
1. A
2. B
3. C

Once I have your answers, I will generate a detailed technical specification.""",
                tool_calls=[],
                citations=[],
            )
    
    def _generate_fallback_mcq_questions(self, user_request: str, repo_context: dict) -> str:
        """Generate fallback MCQ clarifying questions if LLM generation fails."""
        questions = []
        
        # Question 1: Architecture/Integration
        if repo_context.get("framework"):
            q1 = f"""1. I see this project uses {repo_context['framework']}. How should this feature integrate with the existing architecture?
   A. Integrate directly into existing codebase following current patterns
   B. Create new module/service that follows existing architecture
   C. Build as separate microservice
   D. Extend existing components with new functionality"""
        else:
            q1 = """1. How should this feature integrate with the existing codebase architecture?
   A. Integrate directly into existing codebase
   B. Create new module/service
   C. Build as separate microservice
   D. Extend existing components"""
        questions.append(q1)
        
        # Question 2: Database/Data layer
        if repo_context.get("database"):
            q2 = f"""2. The project uses {repo_context['database']}. How should data be handled?
   A. Use existing database schema/tables
   B. Create new tables/models in existing database
   C. Use separate database instance
   D. No persistent storage needed"""
        else:
            q2 = """2. What data persistence approach should be used?
   A. Use existing database if available
   B. Create new database schema
   C. Use in-memory storage only
   D. External data service"""
        questions.append(q2)
        
        # Question 3: API/Interface
        q3 = """3. How should this feature expose its interface?
   A. REST API endpoints with existing authentication
   B. GraphQL API
   C. WebSocket/real-time connections
   D. Internal service calls only (no external API)"""
        questions.append(q3)
        
        # Question 4: Authentication/Authorization
        q4 = """4. What authentication/authorization is required?
   A. Use existing authentication system
   B. Implement new authentication
   C. No authentication needed (public/internal)
   D. Third-party auth provider (OAuth, etc.)"""
        questions.append(q4)
        
        # Question 5: Deployment/Environment
        q5 = """5. Are there deployment or environment considerations?
   A. Deploy alongside existing services
   B. Separate deployment pipeline
   C. Containerized/microservice deployment
   D. Serverless/cloud function deployment"""
        questions.append(q5)
        
        return "\n\n".join(questions[:5])  # Ensure max 5
    
    async def _extract_clarification_answers(self, history: list[str]) -> str:
        """
        Extract user's clarification answers from conversation history.
        
        Parses MCQ responses (like "1. A, 2. B" or "1: A\n2: B") and formats them.
        Returns formatted answers with question context.
        """
        if not history or len(history) < 2:
            return ""
        
        # Find the last assistant message with clarification questions
        import re
        clarification_questions_msg = None
        clarification_idx = None
        
        for i in range(len(history) - 1, -1, -1):
            msg = history[i]
            if "Before generating" in msg or ("clarification" in msg.lower() and ("A." in msg or "B." in msg)):
                clarification_questions_msg = msg
                clarification_idx = i
                break
        
        if not clarification_questions_msg or clarification_idx is None:
            return ""
        
        # Check if next message is user response
        if clarification_idx + 1 >= len(history):
            return ""
        
        user_response = history[clarification_idx + 1]
        # Remove "User:" prefix if present
        if user_response.startswith("User:"):
            user_response = user_response[5:].strip()
        
        # Extract questions from the clarification message
        question_pattern = r'\n?\d+\.\s+([^\n]+(?:\n\s+[A-D]\.\s+[^\n]+)*)'
        questions = re.findall(question_pattern, clarification_questions_msg, re.MULTILINE)
        
        # Parse user's answers (handle formats like "1. A", "1: A", "1 A", etc.)
        answer_patterns = [
            r'(\d+)[\.:]\s*([A-D])',  # "1. A" or "1: A"
            r'(\d+)\s+([A-D])',       # "1 A"
            r'Question\s+(\d+)[\.:]?\s*([A-D])',  # "Question 1: A"
        ]
        
        answers = {}
        for pattern in answer_patterns:
            matches = re.findall(pattern, user_response, re.IGNORECASE)
            for match in matches:
                q_num = int(match[0])
                answer = match[1].upper()
                answers[q_num] = answer
        
        # If no structured answers found, try to extract from lines
        if not answers:
            lines = user_response.split('\n')
            for line in lines:
                line = line.strip()
                for pattern in answer_patterns:
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        q_num = int(match.group(1))
                        answer = match.group(2).upper()
                        answers[q_num] = answer
        
        # Format answers with question context
        formatted_answers = []
        for i, question in enumerate(questions[:5], 1):
            if i in answers:
                formatted_answers.append(f"Question {i}: Selected option {answers[i]}")
            else:
                # Try to infer from user response text
                question_lower = question.lower()
                if any(opt in user_response.lower() for opt in ['option a', 'choice a', 'answer a']):
                    formatted_answers.append(f"Question {i}: Selected option A (inferred)")
                elif any(opt in user_response.lower() for opt in ['option b', 'choice b', 'answer b']):
                    formatted_answers.append(f"Question {i}: Selected option B (inferred)")
                else:
                    formatted_answers.append(f"Question {i}: Answer not clearly specified")
        
        if formatted_answers:
            return "\n".join(formatted_answers) + f"\n\nFull user response: {user_response}"
        
        return user_response  # Fallback to raw response
    
    async def _enriched_context(self, ctx: ChatContext) -> ChatContext:
        """Enrich context with conversation state and Q&A information."""
        # Add conversation history analysis to context
        if ctx.history:
            previous_spec = await self._extract_previous_spec(ctx.history)
            if previous_spec:
                ctx.additional_context += f"\n\nPREVIOUS SPECIFICATION EXISTS IN CONVERSATION HISTORY\n"
                ctx.additional_context += f"Previous spec summary: {previous_spec.tl_dr[:200]}...\n"
        
        return ctx
    
    async def run(self, ctx: ChatContext) -> ChatAgentResponse:
        """Run conversational spec generation with iterative refinement support."""
        # Enrich context with conversation state
        ctx = await self._enriched_context(ctx)
        
        # Classify the request
        request_type, target_sections = await self._classify_request(ctx.query, ctx.history)
        previous_spec = await self._extract_previous_spec(ctx.history)
        
        logger.info(f"[SPEC_GEN] Request classified as: {request_type}, target_sections: {target_sections}")
        
        # Handle clear partial updates with specific sections
        if request_type == 'partial' and previous_spec and target_sections:
            try:
                logger.info(f"[SPEC_GEN] Executing partial workflow for: {target_sections}")
                specification = await self._execute_partial_workflow(
                    target_sections=target_sections,
                    previous_spec=previous_spec,
                    user_request=ctx.query,
                    qa_context=ctx.additional_context,
                    project_id=ctx.project_id,
                    project_name=ctx.project_name,
                    node_ids=ctx.node_ids,
                )
                formatted_spec = self._format_specification(specification)
                return ChatAgentResponse(
                    response=f"# Specification Updated\n\nI've updated the following sections: {', '.join(target_sections)}\n\n{formatted_spec}",
                    tool_calls=[],
                    citations=[],
                )
            except Exception as e:
                logger.exception(f"Error in partial workflow execution: {e}")
                # Fall back to conversational agent
                logger.info("[SPEC_GEN] Falling back to conversational agent for partial update")
                agent = self._build_agent()
                return await agent.run(ctx)
        elif request_type == 'partial' and not previous_spec:
            logger.warning("[SPEC_GEN] Partial update requested but no previous spec found in history")
            # Fall back to conversational agent to ask for clarification
            agent = self._build_agent()
            return await agent.run(ctx)
        
        # Handle new requests - check if clarification is needed
        if request_type == 'new':
            # Check if clarification has been completed
            clarification_completed = await self._check_clarification_completed(ctx.history)
            
            if not clarification_completed:
                # Enter clarification phase
                logger.info("[SPEC_GEN] Clarification phase required for new specification")
                return await self._clarification_phase(ctx)
            else:
                # Clarification completed, proceed with spec generation
                logger.info("[SPEC_GEN] Clarification completed, executing full workflow for new specification")
                
                # Extract clarification answers
                clarification_answers = await self._extract_clarification_answers(ctx.history)
                
                # Combine original request with clarification answers
                combined_qa_context = ctx.additional_context
                if clarification_answers:
                    combined_qa_context += f"\n\nCLARIFICATION ANSWERS:\n{clarification_answers}\n"
                
                try:
                    specification = await self._generate_specification(
                        user_prompt=ctx.query,
                        qa_context=combined_qa_context,
                        project_id=ctx.project_id,
                        project_name=ctx.project_name,
                        node_ids=ctx.node_ids,
                    )
                    formatted_spec = self._format_specification(specification)
                    return ChatAgentResponse(
                        response=f"# Specification Generated\n\nI've generated a complete technical specification based on your request and clarification answers.\n\n{formatted_spec}",
                        tool_calls=[],
                        citations=[],
                    )
                except Exception as e:
                    logger.exception(f"Error generating specification: {e}")
                    # Provide helpful error message and fall back to conversational agent
                    error_msg = str(e)
                    if "success_metrics" in error_msg.lower() or "validation" in error_msg.lower():
                        logger.warning("[SPEC_GEN] Validation error detected, falling back to conversational agent for recovery")
                    else:
                        logger.warning("[SPEC_GEN] Workflow error detected, falling back to conversational agent")
                    # Fall back to conversational agent for error handling
                    agent = self._build_agent()
                    return await agent.run(ctx)
        
        # For refinements, clarifications, or ambiguous requests - use conversational agent
        # The agent will ask clarifying questions or understand what needs to be changed
        logger.info("[SPEC_GEN] Using conversational agent for refinement/clarification")
        agent = self._build_agent()
        response = await agent.run(ctx)
        
        # If conversational agent didn't generate a spec but we have a previous spec and clear intent,
        # we could execute workflow here, but for now let the agent handle it conversationally
        return response
    
    async def run_stream(
        self, ctx: ChatContext
    ) -> AsyncGenerator[ChatAgentResponse, None]:
        """Run conversational spec generation with streaming.
        
        Follows the same workflow logic as run() (classification, clarification,
        research, 7-step generation) but uses streaming for the wrapper agent fallback.
        """
        # Enrich context with conversation state
        ctx = await self._enriched_context(ctx)
        
        # Classify the request
        request_type, target_sections = await self._classify_request(ctx.query, ctx.history)
        previous_spec = await self._extract_previous_spec(ctx.history)
        
        logger.info(f"[SPEC_GEN_STREAM] Request classified as: {request_type}, target_sections: {target_sections}")
        
        # Handle clear partial updates with specific sections
        if request_type == 'partial' and previous_spec and target_sections:
            try:
                logger.info(f"[SPEC_GEN_STREAM] Executing partial workflow for: {target_sections}")
                specification = await self._execute_partial_workflow(
                    target_sections=target_sections,
                    previous_spec=previous_spec,
                    user_request=ctx.query,
                    qa_context=ctx.additional_context,
                    project_id=ctx.project_id,
                    project_name=ctx.project_name,
                    node_ids=ctx.node_ids,
                )
                formatted_spec = self._format_specification(specification)
                yield ChatAgentResponse(
                    response=f"# Specification Updated\n\nI've updated the following sections: {', '.join(target_sections)}\n\n{formatted_spec}",
                    tool_calls=[],
                    citations=[],
                )
                return
            except Exception as e:
                logger.exception(f"Error in partial workflow execution (stream): {e}")
                logger.info("[SPEC_GEN_STREAM] Falling back to conversational agent for partial update")
                agent = self._build_agent()
                async for chunk in agent.run_stream(ctx):
                    yield chunk
                return
        elif request_type == 'partial' and not previous_spec:
            logger.warning("[SPEC_GEN_STREAM] Partial update requested but no previous spec found in history")
            agent = self._build_agent()
            async for chunk in agent.run_stream(ctx):
                yield chunk
            return
        
        # Handle new requests - check if clarification is needed
        if request_type == 'new':
            clarification_completed = await self._check_clarification_completed(ctx.history)
            
            if not clarification_completed:
                # Enter clarification phase — stream so user sees tool calls
                logger.info("[SPEC_GEN_STREAM] Clarification phase required for new specification")
                async for chunk in self._clarification_phase_stream(ctx):
                    yield chunk
                return
            else:
                # Clarification completed, proceed with spec generation
                logger.info("[SPEC_GEN_STREAM] Clarification completed, executing full workflow for new specification")
                
                # Extract clarification answers
                clarification_answers = await self._extract_clarification_answers(ctx.history)
                
                # Combine original request with clarification answers
                combined_qa_context = ctx.additional_context
                if clarification_answers:
                    combined_qa_context += f"\n\nCLARIFICATION ANSWERS:\n{clarification_answers}\n"
                
                try:
                    specification = await self._generate_specification(
                        user_prompt=ctx.query,
                        qa_context=combined_qa_context,
                        project_id=ctx.project_id,
                        project_name=ctx.project_name,
                        node_ids=ctx.node_ids,
                    )
                    formatted_spec = self._format_specification(specification)
                    yield ChatAgentResponse(
                        response=f"# Specification Generated\n\nI've generated a complete technical specification based on your request and clarification answers.\n\n{formatted_spec}",
                        tool_calls=[],
                        citations=[],
                    )
                    return
                except Exception as e:
                    logger.exception(f"Error generating specification (stream): {e}")
                    logger.warning("[SPEC_GEN_STREAM] Workflow error detected, falling back to conversational agent")
                    agent = self._build_agent()
                    async for chunk in agent.run_stream(ctx):
                        yield chunk
                    return
        
        # Handle refinement or other request types - use the conversational wrapper agent with streaming
        agent = self._build_agent()
        async for chunk in agent.run_stream(ctx):
            yield chunk
