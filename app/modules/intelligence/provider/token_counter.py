"""
Token Counter Module

This module provides token counting and cost calculation for LLM API calls.
It supports multiple models and provides accurate cost estimates.
"""

import tiktoken
from typing import List, Dict, Optional


class TokenCounter:
    """
    Count tokens and calculate costs for different LLM models.
    
    This class uses tiktoken for accurate token counting (for OpenAI models)
    and provides cost calculations based on current pricing.
    
    Example usage:
        counter = TokenCounter("gpt-4")
        
        # Count tokens in text
        token_count = counter.count_tokens("Hello, world!")
        
        # Count tokens in messages
        messages = [
            {"role": "user", "content": "What is Python?"}
        ]
        msg_tokens = counter.count_message_tokens(messages)
        
        # Calculate cost
        cost = counter.calculate_cost(input_tokens=100, output_tokens=50)
        print(f"Cost: ${cost:.4f}")
    """
    
    # Token pricing per 1,000 tokens (as of Dec 2025)
    # Update these values as pricing changes
    PRICING = {
        'gpt-4': {'input': 0.03, 'output': 0.06},
        'gpt-4-turbo': {'input': 0.01, 'output': 0.03},
        'gpt-3.5-turbo': {'input': 0.0015, 'output': 0.002},
        'claude-3-opus': {'input': 0.015, 'output': 0.075},
        'claude-3-sonnet': {'input': 0.003, 'output': 0.015},
        'claude-3-haiku': {'input': 0.00025, 'output': 0.00125},
    }
    
    def __init__(self, model: str):
        """
        Initialize the token counter for a specific model.
        
        Args:
            model: The model name (e.g., "gpt-4", "claude-3-opus")
        """
        self.model = model
        
        # Initialize tokenizer (use tiktoken for OpenAI models)
        try:
            if 'gpt' in model.lower():
                # Use model-specific encoding
                self.encoding = tiktoken.encoding_for_model(model)
            else:
                # Use cl100k_base as approximation for other models
                self.encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            # Fallback to cl100k_base if model not found
            self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in a text string.
        
        Args:
            text: The text to count tokens for
            
        Returns:
            int: Number of tokens in the text
        """
        if not text:
            return 0
        return len(self.encoding.encode(text))
    
    def count_message_tokens(self, messages: List[Dict[str, str]]) -> int:
        """
        Count tokens in a list of messages (chat format).
        
        This accounts for message formatting overhead that the API adds.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            
        Returns:
            int: Total token count including formatting overhead
        """
        if not messages:
            return 0
        
        total = 0
        
        # Count tokens for each message
        for message in messages:
            # Add tokens for role
            role = message.get('role', '')
            total += self.count_tokens(role)
            
            # Add tokens for content
            content = message.get('content', '')
            total += self.count_tokens(content)
            
            # Add message formatting overhead (4 tokens per message for OpenAI)
            total += 4
        
        # Add conversation-level formatting (2 tokens)
        total += 2
        
        return total
    
    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Calculate the cost in USD for a given number of tokens.
        
        Args:
            input_tokens: Number of input (prompt) tokens
            output_tokens: Number of output (completion) tokens
            
        Returns:
            float: Total cost in USD
        """
        # Normalize model name for pricing lookup
        model_key = self._normalize_model_name(self.model)
        
        # Get pricing (use default if model not found)
        pricing = self.PRICING.get(
            model_key, 
            {'input': 0.01, 'output': 0.02}  # Conservative default
        )
        
        # Calculate costs
        input_cost = (input_tokens / 1000) * pricing['input']
        output_cost = (output_tokens / 1000) * pricing['output']
        
        return input_cost + output_cost
    
    def _normalize_model_name(self, model: str) -> str:
        """
        Normalize model name for pricing lookup.
        
        Args:
            model: The model name
            
        Returns:
            str: Normalized model name that matches PRICING keys
        """
        model_lower = model.lower()
        
        # Check each pricing key
        for key in self.PRICING.keys():
            if key in model_lower:
                return key
        
        # Default to gpt-3.5-turbo pricing if no match
        return 'gpt-3.5-turbo'
    
    def get_pricing_info(self) -> Dict[str, float]:
        """
        Get pricing information for the current model.
        
        Returns:
            Dict with 'input' and 'output' pricing per 1K tokens
        """
        model_key = self._normalize_model_name(self.model)
        return self.PRICING.get(
            model_key,
            {'input': 0.01, 'output': 0.02}
        )
