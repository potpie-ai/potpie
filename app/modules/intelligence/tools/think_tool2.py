import asyncio
from typing import Dict, Any, List, Optional
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from app.modules.intelligence.provider.provider_service import ProviderService

# DSPy for structured reasoning
import dspy
from dspy import Module, ChainOfThought, Signature, InputField, OutputField


class ThinkToolInput(BaseModel):
    thought: str = Field(description="Your thoughts to process")


class AnalyzeThought(Signature):
    """Analyze a thought and break it down into structured components."""

    thought = InputField(desc="The thought or problem to analyze")
    problem_type = OutputField(
        desc="Type of problem (bug_fix, test_failure, feature_request, planning, etc.)"
    )
    key_components = OutputField(desc="Main components or aspects of the problem")
    constraints = OutputField(desc="Constraints, limitations, or rules that apply")
    required_info = OutputField(desc="Information needed to proceed")
    available_info = OutputField(desc="Information already available")


class GenerateReasoningSteps(Signature):
    """Generate structured reasoning steps for a problem."""

    problem_type = InputField(desc="Type of problem being solved")
    key_components = InputField(desc="Main components of the problem")
    constraints = InputField(desc="Applicable constraints and limitations")
    reasoning_steps = OutputField(desc="Step-by-step reasoning approach")
    considerations = OutputField(desc="Important considerations for each step")


class EvaluateSolutions(Signature):
    """Evaluate potential solutions and recommend the best approach."""

    problem_type = InputField(desc="Type of problem")
    reasoning_steps = InputField(desc="Reasoning steps taken")
    solutions = InputField(desc="Potential solutions identified")
    evaluation = OutputField(
        desc="Evaluation of each solution (pros, cons, feasibility)"
    )
    recommendation = OutputField(desc="Recommended solution with rationale")
    next_actions = OutputField(desc="Specific next actions to take")


class StructuredThinker(Module):
    """DSPy module for structured thinking and problem analysis."""

    def __init__(self):
        super().__init__()
        self.analyze = ChainOfThought(AnalyzeThought)
        self.generate_steps = ChainOfThought(GenerateReasoningSteps)
        self.evaluate = ChainOfThought(EvaluateSolutions)

    def forward(self, thought: str) -> Dict[str, Any]:
        """Process a thought through structured reasoning."""

        # Step 1: Analyze the thought
        analysis = self.analyze(thought=thought)

        # Step 2: Generate reasoning steps
        reasoning = self.generate_steps(
            problem_type=analysis.problem_type,
            key_components=analysis.key_components,
            constraints=analysis.constraints,
        )

        # Step 3: Generate and evaluate solutions
        # First, let's identify potential solutions based on the reasoning
        solutions_prompt = f"""
        Based on the problem type '{analysis.problem_type}' and reasoning steps:
        {reasoning.reasoning_steps}
        
        Generate 3-5 potential solutions or approaches.
        """

        evaluation = self.evaluate(
            problem_type=analysis.problem_type,
            reasoning_steps=reasoning.reasoning_steps,
            solutions=solutions_prompt,
        )

        return {
            "analysis": {
                "problem_type": analysis.problem_type,
                "key_components": analysis.key_components,
                "constraints": analysis.constraints,
                "required_info": analysis.required_info,
                "available_info": analysis.available_info,
            },
            "reasoning": {
                "steps": reasoning.reasoning_steps,
                "considerations": reasoning.considerations,
            },
            "evaluation": {
                "solution_analysis": evaluation.evaluation,
                "recommendation": evaluation.recommendation,
                "next_actions": evaluation.next_actions,
            },
        }


class CustomDSPyLM(dspy.LM):
    """Custom DSPy Language Model that integrates with ProviderService."""

    def __init__(self, provider_service: ProviderService):
        self.provider_service = provider_service
        # Initialize with a generic model name
        super().__init__(model="custom-provider")

    def basic_request(self, prompt: str, **kwargs) -> List[str]:
        """Basic request method required by DSPy LM interface."""
        try:
            # Create message format expected by provider service
            messages = [{"role": "user", "content": prompt}]

            # Run the async call in a synchronous context
            response = self._run_async_call(messages)
            return [response] if isinstance(response, str) else response

        except Exception as e:
            print(f"DSPy LM request failed: {e}")
            # Return a fallback response
            return [f"Error in LM request: {str(e)}"]

    def _run_async_call(self, messages: List[Dict[str, str]]) -> str:
        """Helper to run async call in sync context."""
        try:
            # Check if we're already in an event loop
            try:
                loop = asyncio.get_running_loop()
                # We're in an async context, use thread executor
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        lambda: asyncio.run(self._async_call(messages))
                    )
                    return future.result(timeout=30)  # 30 second timeout
            except RuntimeError:
                # No running loop, we can create one
                return asyncio.run(self._async_call(messages))
        except Exception as e:
            raise Exception(f"Failed to execute async call: {e}")

    async def _async_call(self, messages: List[Dict[str, str]]) -> str:
        """Async helper method to call the provider service."""
        try:
            response = await self.provider_service.call_llm(
                messages=messages, config_type="chat"
            )
            return response if isinstance(response, str) else str(response)
        except Exception as e:
            raise Exception(f"Provider service call failed: {e}")

    def __call__(
        self, prompt: str = None, messages: List = None, **kwargs
    ) -> List[str]:
        """Make the class callable with DSPy expected signature."""
        # Handle different ways DSPy might call this
        if prompt is not None:
            return self.basic_request(prompt, **kwargs)
        elif messages is not None:
            # If messages are provided directly
            if isinstance(messages, list) and len(messages) > 0:
                if isinstance(messages[0], dict):
                    # Extract content from message format
                    prompt = messages[-1].get("content", str(messages))
                else:
                    prompt = str(messages)
                return self.basic_request(prompt, **kwargs)

        # Fallback - try to extract from kwargs
        if "prompt" in kwargs:
            return self.basic_request(kwargs["prompt"], **kwargs)

        # If we get here, we don't have enough information
        print(
            f"DSPy LM called with unexpected arguments: prompt={prompt}, messages={messages}, kwargs={kwargs}"
        )
        return ["Unable to process request - missing prompt"]

    def generate(self, prompt: str = None, **kwargs) -> List[str]:
        """Alternative method name that DSPy might use."""
        if prompt is None and "prompt" in kwargs:
            prompt = kwargs["prompt"]
        return self.basic_request(prompt or "", **kwargs)

    # Additional methods that DSPy might expect
    def request(self, prompt: str, **kwargs) -> List[str]:
        """Request method for DSPy compatibility."""
        return self.basic_request(prompt, **kwargs)

    def chat(self, messages: List[Dict], **kwargs) -> List[str]:
        """Chat method for message-based interactions."""
        if messages and len(messages) > 0:
            prompt = messages[-1].get("content", "")
            return self.basic_request(prompt, **kwargs)
        return ["No messages provided"]


class FallbackReasoner:
    """Fallback reasoner for when DSPy is not available or fails."""

    def reason(self, thought: str) -> Dict[str, Any]:
        """Simple rule-based reasoning as fallback."""

        # Basic keyword analysis
        thought_lower = thought.lower()

        # Determine problem type
        if any(
            word in thought_lower
            for word in ["bug", "error", "issue", "problem", "broken"]
        ):
            problem_type = "bug_fix"
        elif any(word in thought_lower for word in ["test", "failing", "failure"]):
            problem_type = "test_failure"
        elif any(
            word in thought_lower for word in ["feature", "add", "implement", "new"]
        ):
            problem_type = "feature_request"
        elif any(word in thought_lower for word in ["plan", "strategy", "approach"]):
            problem_type = "planning"
        else:
            problem_type = "general_analysis"

        # Generate structured analysis
        return {
            "analysis": {
                "problem_type": problem_type,
                "key_components": self._extract_components(thought),
                "constraints": self._extract_constraints(thought),
                "required_info": self._extract_required_info(thought),
                "available_info": [
                    thought[:200] + "..." if len(thought) > 200 else thought
                ],
            },
            "reasoning": {
                "steps": self._generate_steps(problem_type),
                "considerations": self._generate_considerations(problem_type),
            },
            "evaluation": {
                "solution_analysis": self._analyze_solutions(problem_type, thought),
                "recommendation": self._generate_recommendation(problem_type),
                "next_actions": self._generate_next_actions(problem_type),
            },
        }

    def _extract_components(self, thought: str) -> str:
        sentences = [s.strip() for s in thought.split(".") if s.strip()]
        key_phrases = []

        # Look for important phrases
        important_words = [
            "implement",
            "fix",
            "create",
            "update",
            "test",
            "deploy",
            "analyze",
        ]
        for sentence in sentences:
            for word in important_words:
                if word in sentence.lower():
                    key_phrases.append(sentence[:100])
                    break

        if key_phrases:
            return "; ".join(key_phrases[:3])
        else:
            return f"Identified {len(sentences)} main points in the problem description"

    def _extract_constraints(self, thought: str) -> str:
        constraint_words = [
            ("cannot", "Cannot constraints"),
            ("must not", "Prohibition constraints"),
            ("limited", "Resource constraints"),
            ("deadline", "Time constraints"),
            ("budget", "Budget constraints"),
            ("requirement", "Requirement constraints"),
        ]

        found_constraints = []
        thought_lower = thought.lower()

        for word, category in constraint_words:
            if word in thought_lower:
                found_constraints.append(category)

        if found_constraints:
            return f"Identified constraints: {', '.join(found_constraints)}"
        else:
            return "No explicit constraints mentioned - consider standard development constraints"

    def _extract_required_info(self, thought: str) -> List[str]:
        thought_lower = thought.lower()
        required_info = []

        if "bug" in thought_lower or "error" in thought_lower:
            required_info.extend(
                [
                    "Error logs and stack traces",
                    "Steps to reproduce the issue",
                    "Expected vs actual behavior",
                ]
            )
        elif "test" in thought_lower:
            required_info.extend(
                [
                    "Test failure details",
                    "Test environment information",
                    "Recent code changes",
                ]
            )
        elif "feature" in thought_lower:
            required_info.extend(
                [
                    "Detailed requirements",
                    "User stories or use cases",
                    "Acceptance criteria",
                ]
            )
        else:
            required_info.extend(
                ["Complete problem context", "Success criteria", "Available resources"]
            )

        return required_info

    def _generate_steps(self, problem_type: str) -> str:
        steps_map = {
            "bug_fix": """
1. Reproduce the issue consistently
2. Identify the root cause through debugging
3. Design a minimal fix that addresses the root cause
4. Test the fix thoroughly in multiple scenarios
5. Review and deploy the solution
            """.strip(),
            "test_failure": """
1. Analyze the failing test(s) and error messages
2. Check for environmental issues or dependencies
3. Review recent code changes that might affect tests
4. Fix the underlying issues
5. Verify all tests pass and run regression testing
            """.strip(),
            "feature_request": """
1. Define clear requirements and acceptance criteria
2. Design the solution architecture
3. Break down implementation into manageable tasks
4. Implement core functionality with tests
5. Review, refine, and deploy
            """.strip(),
            "planning": """
1. Break down the problem into smaller components
2. Identify dependencies and constraints
3. Estimate effort and resources needed
4. Create a timeline with milestones
5. Plan for monitoring and evaluation
            """.strip(),
            "general_analysis": """
1. Break down the problem systematically
2. Gather all relevant information
3. Analyze different approaches and options
4. Evaluate trade-offs and risks
5. Make informed decisions and plan implementation
            """.strip(),
        }
        return steps_map.get(problem_type, steps_map["general_analysis"])

    def _generate_considerations(self, problem_type: str) -> List[str]:
        base_considerations = [
            "Verify all assumptions before proceeding",
            "Consider edge cases and error handling",
            "Think about the impact on existing systems",
            "Plan for testing and validation",
        ]

        type_specific = {
            "bug_fix": [
                "Don't introduce new bugs",
                "Minimize code changes",
                "Consider backwards compatibility",
            ],
            "test_failure": [
                "Check test environment",
                "Verify test data",
                "Consider flaky tests",
            ],
            "feature_request": [
                "User experience impact",
                "Performance implications",
                "Maintainability",
            ],
            "planning": [
                "Resource availability",
                "Timeline feasibility",
                "Risk mitigation",
            ],
        }

        return base_considerations + type_specific.get(problem_type, [])

    def _analyze_solutions(self, problem_type: str, thought: str) -> str:
        return f"""
For {problem_type} problems, consider multiple approaches:
- Quick fixes vs comprehensive solutions
- Risk vs benefit analysis
- Resource requirements and timeline
- Long-term maintainability
- Impact on stakeholders

Based on the problem description, evaluate each potential solution against these criteria.
        """.strip()

    def _generate_recommendation(self, problem_type: str) -> str:
        recommendations = {
            "bug_fix": "Prioritize a minimal, well-tested fix that addresses the root cause without introducing new risks",
            "test_failure": "Focus on identifying and fixing the underlying cause rather than just making tests pass",
            "feature_request": "Start with a minimal viable implementation and iterate based on feedback",
            "planning": "Break down into smaller, manageable phases with clear deliverables and checkpoints",
            "general_analysis": "Take a systematic approach, gathering information before making decisions",
        }
        return recommendations.get(problem_type, recommendations["general_analysis"])

    def _generate_next_actions(self, problem_type: str) -> List[str]:
        actions_map = {
            "bug_fix": [
                "Gather detailed error information",
                "Set up debugging environment",
                "Create minimal reproduction case",
            ],
            "test_failure": [
                "Review test logs and error messages",
                "Check test environment configuration",
                "Identify recent changes that might affect tests",
            ],
            "feature_request": [
                "Clarify requirements with stakeholders",
                "Create technical design document",
                "Set up development environment",
            ],
            "planning": [
                "List all known requirements and constraints",
                "Identify key stakeholders to consult",
                "Create initial project timeline",
            ],
            "general_analysis": [
                "Gather additional context and information",
                "Consult with relevant experts or stakeholders",
                "Document findings and create action plan",
            ],
        }
        return actions_map.get(problem_type, actions_map["general_analysis"])


class ThinkTool:
    """Enhanced tool for thinking and processing thoughts using DSPy."""

    name = "think"
    description = """Use this tool to think about something systematically. It will analyze the problem, generate structured reasoning steps, and provide recommendations. Use it when complex reasoning or brainstorming is needed. For example, if you explore the repo and discover the source of a bug, call this tool to systematically analyze the bug, brainstorm solutions, and assess which approach is likely to be most effective."""

    def __init__(self, sql_db: Session, user_id: str):
        self.sql_db = sql_db
        self.user_id = user_id
        self.provider_service = ProviderService(sql_db, user_id)

        # Initialize structured reasoner
        self.dspy_configured = False
        self.structured_thinker = None

        try:
            self.dspy_configured = self._configure_dspy()
            if self.dspy_configured:
                self.structured_thinker = StructuredThinker()
                print("DSPy configured successfully")
            else:
                print("DSPy configuration failed, using fallback")
        except Exception as e:
            print(f"DSPy configuration error: {e}")
            self.dspy_configured = False
            self.structured_thinker = None

        # Fallback reasoner
        self.fallback_reasoner = FallbackReasoner()

    def _configure_dspy(self) -> bool:
        """Configure DSPy to use your existing LLM provider."""
        try:
            # Check if the provider service is properly configured
            if not self.provider_service:
                print("Provider service not available")
                return False

            # Create and configure custom DSPy LM
            custom_lm = CustomDSPyLM(self.provider_service)

            # Configure DSPy to use our custom LM
            dspy.configure(lm=custom_lm)

            # Test the configuration with a simple prompt
            try:
                test_response = custom_lm("Test prompt for DSPy configuration")
                if test_response and len(test_response) > 0:
                    print(f"DSPy LM test successful: {test_response[0][:50]}...")
                    return True
                else:
                    print("DSPy LM test returned empty response")
                    return False
            except Exception as e:
                print(f"DSPy LM test failed: {e}")
                return False

        except Exception as e:
            print(f"Failed to configure DSPy: {e}")
            return False

    async def arun(self, thought: str) -> Dict[str, Any]:
        """Process the thought using structured thinking asynchronously."""
        try:
            # Try structured reasoning first
            if self.dspy_configured and self.structured_thinker:
                try:
                    print("Attempting structured reasoning with DSPy...")
                    # Run DSPy in a way that handles async context properly
                    structured_result = await self._run_dspy_reasoning(thought)

                    # Format the result for better presentation
                    analysis = self._format_structured_analysis(structured_result)

                    return {
                        "success": True,
                        "method": "structured_reasoning_dspy",
                        "analysis": analysis,
                        "raw_structured_output": structured_result,
                    }

                except Exception as e:
                    print(f"Structured reasoning failed, falling back: {e}")

            # Fallback to rule-based reasoning
            print("Using fallback reasoning...")
            fallback_result = self.fallback_reasoner.reason(thought)
            fallback_analysis = self._format_fallback_analysis(fallback_result)

            return {
                "success": True,
                "method": "fallback_reasoning",
                "analysis": fallback_analysis,
                "structured_components": fallback_result,
            }

        except Exception as e:
            # Final fallback to original LLM-based approach
            print(f"All structured approaches failed, using LLM fallback: {e}")
            return await self._original_llm_approach(thought, str(e))

    async def _run_dspy_reasoning(self, thought: str) -> Dict[str, Any]:
        """Run DSPy reasoning in a way that handles async context."""
        try:
            # Run DSPy in thread executor to avoid async issues
            import concurrent.futures

            def run_dspy():
                try:
                    return self.structured_thinker(thought)
                except Exception as e:
                    print(f"Error in DSPy forward pass: {e}")
                    raise

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_dspy)
                result = future.result(timeout=90)  # 90 second timeout
                return result

        except Exception as e:
            print(f"DSPy execution failed: {e}")
            raise

    def _format_structured_analysis(self, result: Dict[str, Any]) -> str:
        """Format DSPy structured reasoning results."""
        analysis = result["analysis"]
        reasoning = result["reasoning"]
        evaluation = result["evaluation"]

        return f"""
## Structured Analysis (DSPy)

### Problem Analysis
- **Type**: {analysis['problem_type']}
- **Key Components**: {analysis['key_components']}
- **Constraints**: {analysis['constraints']}
- **Required Info**: {analysis['required_info']}
- **Available Info**: {analysis['available_info']}

### Reasoning Steps
{reasoning['steps']}

**Considerations**: {reasoning['considerations']}

### Solution Evaluation
**Analysis**: {evaluation['solution_analysis']}

**Recommendation**: {evaluation['recommendation']}

**Next Actions**: {evaluation['next_actions']}

---
*Analysis generated using structured reasoning with DSPy*
        """.strip()

    def _format_fallback_analysis(self, result: Dict[str, Any]) -> str:
        """Format fallback reasoning results."""
        analysis = result["analysis"]
        reasoning = result["reasoning"]
        evaluation = result["evaluation"]

        return f"""
## Structured Analysis

### Problem Analysis
- **Type**: {analysis['problem_type']}
- **Key Components**: {analysis['key_components']}
- **Constraints**: {analysis['constraints']}
- **Required Information**: {chr(10).join(f"  • {info}" for info in analysis['required_info'])}

### Reasoning Approach
{reasoning['steps']}

**Key Considerations**:
{chr(10).join(f"• {consideration}" for consideration in reasoning['considerations'])}

### Solution Evaluation & Recommendations

{evaluation['solution_analysis']}

**Recommended Approach**: {evaluation['recommendation']}

**Immediate Next Actions**:
{chr(10).join(f"1. {action}" for action in evaluation['next_actions'])}

---
*Analysis generated using enhanced rule-based reasoning*
        """.strip()

    async def _original_llm_approach(
        self, thought: str, error_context: str
    ) -> Dict[str, Any]:
        """Fallback to original LLM-based approach."""
        prompt = f"""
        ## Structured Thinking Analysis
        
        Please provide a systematic analysis of the following thought or problem:
        
        **Problem/Thought**: {thought}
        
        Please structure your response with the following sections:
        
        ### 1. Problem Classification & Analysis
        - Identify the type of problem (bug fix, feature request, planning, etc.)
        - Break down the key components
        - Identify constraints and limitations
        
        ### 2. Information Assessment
        - What information is available?
        - What additional information is needed?
        - What assumptions are being made?
        
        ### 3. Structured Reasoning Approach
        - Provide step-by-step reasoning approach
        - Consider multiple perspectives
        - Identify potential risks and considerations
        
        ### 4. Solution Evaluation
        - Generate multiple potential approaches
        - Evaluate pros and cons of each
        - Consider feasibility and impact
        
        ### 5. Recommendations & Next Steps
        - Provide clear recommendation with rationale
        - List specific, actionable next steps
        - Suggest success criteria and monitoring
        
        Note: Advanced structured reasoning encountered issues ({error_context}), so providing direct expert analysis.
        """

        messages = [
            {
                "role": "system",
                "content": "You are an expert at structured thinking and systematic problem analysis. Provide thorough, well-organized analysis that helps break down complex problems into manageable components.",
            },
            {"role": "user", "content": prompt},
        ]

        try:
            response = await self.provider_service.call_llm(
                messages=messages, config_type="chat"
            )
            return {
                "success": True,
                "method": "llm_fallback",
                "analysis": response,
                "note": "Used enhanced LLM fallback due to structured reasoning issues",
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"All reasoning methods failed: {str(e)}",
                "analysis": "Unable to process thought - please try rephrasing or breaking down the problem.",
            }

    def run(self, thought: str) -> Dict[str, Any]:
        """Synchronous wrapper for arun."""
        try:
            # Check if we're already in an event loop
            try:
                loop = asyncio.get_running_loop()
                # We're in an async context, use thread executor
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(lambda: asyncio.run(self.arun(thought)))
                    return future.result(timeout=120)  # 2 minute timeout
            except RuntimeError:
                # No running loop, we can create one
                return asyncio.run(self.arun(thought))
        except Exception as e:
            print(f"Error in think tool execution: {e}")
            return {
                "success": False,
                "error": str(e),
                "analysis": "Failed to process thought due to execution error.",
            }


def think_tool(sql_db: Session, user_id: str) -> StructuredTool:
    """Create and return the enhanced think tool."""
    tool_instance = ThinkTool(sql_db, user_id)
    return StructuredTool.from_function(
        coroutine=tool_instance.arun,
        func=tool_instance.run,
        name="think",
        description=tool_instance.description,
        args_schema=ThinkToolInput,
    )
