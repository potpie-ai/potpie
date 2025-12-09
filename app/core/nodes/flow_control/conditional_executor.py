"""
Conditional node executor using LLM interface with structured output.

This module provides intelligent conditional evaluation using LLM-based reasoning.
"""

import logging
from typing import Any, Dict, Optional
from pydantic import BaseModel

from app.core.intelligence import LLMInterface
from app.core.executions.state import (
    NodeExecutionContext,
    NodeExecutionResult,
    NodeExecutionStatus,
)
from app.core.nodes.flow_control.conditional import ConditionalEvaluationResult
from app.core.nodes.flow_control.conditional import ConditionalNode

logger = logging.getLogger(__name__)


class ConditionalExecutor:
    """Executor for conditional nodes using LLM-based evaluation."""

    def __init__(self, llm_interface: LLMInterface):
        """
        Initialize the conditional executor.

        Args:
            llm_interface: LLM interface for condition evaluation
        """
        self.llm_interface = llm_interface

    async def execute(
        self, node: ConditionalNode, ctx: NodeExecutionContext
    ) -> NodeExecutionResult:
        """
        Execute a conditional node using LLM-based evaluation.

        Args:
            node: The conditional node to execute
            ctx: Execution context containing variables and state

        Returns:
            NodeExecutionResult with the evaluation result and routing information
        """
        execution_id = ctx.execution_id
        try:
            logger.info(
                f"[execution:{execution_id}] 🔀 Starting conditional node execution for node {node.id}"
            )
            logger.info(
                f"[execution:{execution_id}] 📋 Condition: {node.data.condition}"
            )
            logger.debug(
                f"[execution:{execution_id}] 📄 Previous node result: {ctx.previous_node_result}"
            )

            # Always use LLM evaluation
            logger.info(
                f"[execution:{execution_id}] 🧠 Calling LLM for conditional evaluation..."
            )
            result = await self._evaluate_with_llm(node, ctx)
            logger.info(
                f"[execution:{execution_id}] ✅ LLM evaluation completed: {result}"
            )

            # Create the output with routing information
            output = {
                "condition_result": result.result,
                "should_continue": result.should_continue,
                "reasoning": result.reasoning,
                "confidence": result.confidence,
                "condition": node.data.condition,
                "previous_node_result": ctx.previous_node_result,
            }

            logger.info(
                f"[execution:{execution_id}] 🎯 Conditional evaluation result: result={result.result}, should_continue={result.should_continue}, confidence={result.confidence}"
            )
            logger.debug(f"[execution:{execution_id}] 📄 Reasoning: {result.reasoning}")
            logger.debug(f"[execution:{execution_id}] 📋 Full output: {output}")

            return NodeExecutionResult(
                status=NodeExecutionStatus.COMPLETED,
                output=output,
            )

        except Exception as e:
            logger.error(
                f"[execution:{execution_id}] 💥 Error executing conditional node: {e}"
            )
            logger.exception(f"[execution:{execution_id}] Full traceback:")
            return NodeExecutionResult(
                status=NodeExecutionStatus.FAILED,
                output=f"Conditional evaluation failed: {str(e)}",
            )

    async def _evaluate_with_llm(
        self, node: ConditionalNode, ctx: NodeExecutionContext
    ) -> ConditionalEvaluationResult:
        """
        Evaluate condition using LLM with structured output.

        Args:
            node: The conditional node
            ctx: Execution context

        Returns:
            ConditionalEvaluationResult with structured evaluation
        """
        execution_id = ctx.execution_id
        try:
            logger.info(f"[execution:{execution_id}] 📝 Building evaluation prompt...")
            # Build the prompt for LLM evaluation
            prompt = self._build_evaluation_prompt(node, ctx)
            logger.debug(
                f"[execution:{execution_id}] 📄 Prompt: {prompt[:500]}..."
            )  # Log first 500 chars

            # Prepare context for the LLM
            llm_context = {
                "system_message": self._get_system_message(),
            }

            # Add additional context if provided
            if node.data.llm_context:
                llm_context[
                    "system_message"
                ] += f"\n\nAdditional Context: {node.data.llm_context}"
                logger.debug(
                    f"[execution:{execution_id}] 📋 Added LLM context: {node.data.llm_context}"
                )

            logger.info(
                f"[execution:{execution_id}] 🤖 Making LLM call for conditional evaluation..."
            )
            # Call LLM with structured output
            result = await self.llm_interface.call(
                prompt=prompt,
                output_schema=ConditionalEvaluationResult,
                context=llm_context,
            )
            logger.info(
                f"[execution:{execution_id}] ✅ LLM call completed successfully: {result}"
            )

            return result
        except Exception as e:
            logger.error(f"[execution:{execution_id}] 💥 LLM evaluation failed: {e}")
            logger.exception(
                f"[execution:{execution_id}] Full LLM evaluation traceback:"
            )
            raise

    def _build_evaluation_prompt(
        self, node: ConditionalNode, ctx: NodeExecutionContext
    ) -> str:
        """
        Build the prompt for LLM condition evaluation.

        Args:
            node: The conditional node
            ctx: Execution context

        Returns:
            Formatted prompt string
        """
        # Get available variables from context
        variables = ctx.execution_variables
        previous_result = ctx.previous_node_result

        prompt = f"""
You are evaluating a workflow condition. Please analyze the condition and determine if it evaluates to true or false.

CONDITION TO EVALUATE:
{node.data.condition}

PREVIOUS NODE RESULT (PRIMARY INPUT):
{previous_result}

AVAILABLE VARIABLES (SECONDARY INPUT):
{self._format_variables(variables)}

EVENT PAYLOAD (CONTEXT):
{self._format_event_payload(ctx.event.payload)}

CURRENT ITERATION:
{ctx.current_iteration if hasattr(ctx, 'current_iteration') and ctx.current_iteration is not None else 'Not specified'}

EVALUATION INSTRUCTIONS:
1. PRIMARY FOCUS: Evaluate the condition against the PREVIOUS NODE RESULT
2. SECONDARY: Consider available variables if the condition references them
3. CONTEXT: Use event payload for additional context if needed
4. ITERATION: Consider current iteration count if the condition references iteration logic

The condition should be evaluated primarily based on the previous node's result. Common patterns:
- "previous.success" or "previous.result.success" - check if previous node succeeded
- "previous.failed" or "previous.result.failed" - check if previous node failed
- "previous.output contains X" - check if previous node output contains specific content
- "previous.status == 'completed'" - check previous node status
- "previous.result.count > 5" - check if previous node result count is greater than 5
- "variable.X == previous.result.Y" - compare variables with previous result
- "iteration < 5" or "current_iteration < 5" - check if current iteration is less than 5
- "iteration >= 3" - check if current iteration is greater than or equal to 3

Respond with a structured evaluation including:
- result: true or false (the logical result of the condition)
- should_continue: true or false (whether to continue workflow execution to next nodes)
- reasoning: clear explanation of your evaluation, especially how it relates to the previous node result
- confidence: confidence level (0.0 to 1.0) in your evaluation
"""

        return prompt

    def _get_system_message(self) -> str:
        """Get the system message for LLM evaluation."""
        return """You are a workflow condition evaluator. Your job is to evaluate boolean conditions based on the previous node's result, available variables, and current iteration.

CORE PRINCIPLES:
1. PRIMARY INPUT: Previous node result - this is the main data to evaluate against
2. SECONDARY INPUT: Available variables - use these if the condition references them
3. CONTEXT: Event payload - provides additional context if needed
4. ITERATION: Current iteration count - use this if the condition references iteration logic

EVALUATION GUIDELINES:
1. Always evaluate the condition primarily against the previous node's result
2. Be precise and logical in your evaluation
3. Consider variables only if the condition explicitly references them
4. Consider current iteration if the condition references iteration logic
5. Provide clear reasoning that explains how the condition relates to the previous node result
6. Use confidence levels appropriately:
   - 0.9-1.0: Very confident, clear logical evaluation
   - 0.7-0.8: Confident, but some ambiguity
   - 0.5-0.6: Moderate confidence, multiple interpretations possible
   - 0.3-0.4: Low confidence, significant uncertainty
   - 0.0-0.2: Very low confidence, insufficient information

7. If the condition is ambiguous or unclear, err on the side of false with low confidence
8. Always provide reasoning that explains your evaluation process, particularly how it relates to the previous node result

COMMON CONDITION PATTERNS:
- "previous.success" or "previous.result.success" - check if previous node succeeded
- "previous.failed" or "previous.result.failed" - check if previous node failed
- "previous.output contains X" - check if previous node output contains specific content
- "previous.status == 'completed'" - check previous node status
- "previous.result.count > 5" - check if previous node result count is greater than 5
- "variable.X == previous.result.Y" - compare variables with previous result
- "iteration < 5" or "current_iteration < 5" - check if current iteration is less than 5
- "iteration >= 3" - check if current iteration is greater than or equal to 3

OUTPUT FIELDS:
- result: The logical boolean result of the condition evaluation
- should_continue: Whether the workflow should continue to the next nodes (typically same as result, but can be different for complex routing logic)
- reasoning: Clear explanation of your evaluation, especially how it relates to the previous node result
- confidence: Confidence level in your evaluation (0.0 to 1.0)
"""

    def _format_variables(self, variables: Dict[str, Any]) -> str:
        """Format variables for display in prompt."""
        if not variables:
            return "No variables available"

        formatted = []
        for key, value in variables.items():
            formatted.append(f"  {key}: {value}")
        return "\n".join(formatted)

    def _format_event_payload(self, payload: Dict[str, Any]) -> str:
        """Format event payload for display in prompt."""
        if not payload:
            return "No event payload available"

        try:
            import json

            return json.dumps(payload, indent=2)
        except:
            return str(payload)
