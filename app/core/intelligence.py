from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Any, Dict, Optional
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMInterface(ABC, Generic[T]):
    """
    Simple interface for LLM calls with structured output.

    This interface abstracts away the complexities of model selection,
    retries, and other implementation details, focusing only on the
    core functionality of making a call and getting structured output.
    """

    @abstractmethod
    async def call(
        self,
        prompt: str,
        output_schema: type[T],
        context: Optional[Dict[str, Any]] = None,
    ) -> T:
        """
        Make an LLM call and return structured output.

        Args:
            prompt: The input prompt to send to the LLM
            output_schema: The Pydantic model class defining the expected output structure
            context: Optional additional context (e.g., system messages, examples)

        Returns:
            An instance of the output_schema with the LLM's structured response

        Raises:
            Implementation-specific exceptions for failures
        """
        pass
