from datetime import datetime
from typing import Any, Dict, List, Tuple, Union

from crewai.agents.parser import AgentAction


class FileCallbackHandler:
    def __init__(self, filename: str = "agent_execution_log.md"):
        """Initialize the file callback handler.

        Args:
            filename (str): The markdown file to write the logs to
        """
        self.filename = filename
        # Create or clear the file initially
        with open(self.filename, "w", encoding="utf-8") as f:
            f.write(
                f"# Agent Execution Log\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            )

    def __call__(
        self, step_output: Union[str, List[Tuple[Dict[str, Any], str]], AgentAction]
    ) -> None:
        """Callback function to handle agent execution steps.

        Args:
            step_output: Output from the agent's execution step. Can be:
                        - string
                        - list of (action, observation) tuples
                        - AgentAction from CrewAI
        """
        with open(self.filename, "a", encoding="utf-8") as f:
            f.write(f"\n## Step - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("---\n")

            # Handle AgentAction output
            if isinstance(step_output, AgentAction):
                # Write thought section
                if hasattr(step_output, "thought") and step_output.thought:
                    f.write("### Thought\n")
                    f.write(f"{step_output.thought}\n\n")

                # Write tool section
                if hasattr(step_output, "tool"):
                    f.write("### Action\n")
                    f.write(f"**Tool:** {step_output.tool}\n")

                    # if hasattr(step_output, 'tool_input'):
                    #     try:
                    #         # Try to parse and pretty print JSON input
                    #         tool_input = json.loads(step_output.tool_input)
                    #         formatted_input = json.dumps(tool_input, indent=2)
                    #         f.write(f"**Input:**\n```json\n{formatted_input}\n```\n")
                    #     except (json.JSONDecodeError, TypeError):
                    #         # Fallback to raw string if not JSON
                    #         f.write(f"**Input:**\n```\n{step_output.tool_input}\n```\n")

                # # Write result section
                # if hasattr(step_output, 'result'):
                #     f.write("\n### Result\n")
                #     try:
                #         # Try to parse and pretty print JSON result
                #         result = json.loads(step_output.result)
                #         formatted_result = json.dumps(result, indent=2)
                #         f.write(f"```json\n{formatted_result}\n```\n")
                #     except (json.JSONDecodeError, TypeError):
                #         # Fallback to raw string if not JSON
                #         f.write(f"```\n{step_output.result}\n```\n")

                f.write("\n")
                return

            # Handle single string output
            if isinstance(step_output, str):
                f.write(step_output + "\n")
                return

            for step in step_output:
                if not isinstance(step, tuple):
                    f.write(str(step) + "\n")
                    continue

                action, observation = step

                # Handle action section
                f.write("### Action\n")
                if isinstance(action, dict):
                    if "tool" in action:
                        f.write(f"**Tool:** {action['tool']}\n")
                    if "tool_input" in action:
                        f.write(f"**Input:**\n```\n{action['tool_input']}\n```\n")
                    if "log" in action:
                        f.write(f"**Log:** {action['log']}\n")
                    if "Action" in action:
                        f.write(f"**Action Type:** {action['Action']}\n")
                else:
                    f.write(f"{str(action)}\n")

                # Handle observation section
                f.write("\n### Observation\n")
                if isinstance(observation, str):
                    # Handle special formatting for search-like results
                    lines = observation.split("\n")
                    for line in lines:
                        if line.startswith(("Title:", "Link:", "Snippet:")):
                            key, value = line.split(":", 1)
                            f.write(f"**{key.strip()}:**{value}\n")
                        elif line.startswith("-"):
                            f.write(line + "\n")
                        else:
                            f.write(line + "\n")
                else:
                    f.write(str(observation) + "\n")

                f.write("\n")
