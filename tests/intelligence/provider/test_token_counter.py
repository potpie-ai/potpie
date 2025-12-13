"""
Tests for Token Counter Module
"""

import pytest
from app.modules.intelligence.provider.token_counter import TokenCounter


class TestTokenCounter:
    """Test cases for TokenCounter class"""
    
    def test_initialization(self):
        """Test token counter initialization"""
        counter = TokenCounter("gpt-4")
        assert counter.model == "gpt-4"
        assert counter.encoding is not None
    
    def test_count_simple_tokens(self):
        """Test counting tokens in simple text"""
        counter = TokenCounter("gpt-4")
        
        text = "Hello world"
        token_count = counter.count_tokens(text)
        
        # "Hello world" should be 2 tokens
        assert token_count > 0
        assert token_count <= 5  # Sanity check
    
    def test_count_empty_text(self):
        """Test counting tokens in empty text"""
        counter = TokenCounter("gpt-4")
        
        assert counter.count_tokens("") == 0
        assert counter.count_tokens(None) == 0
    
    def test_count_longer_text(self):
        """Test counting tokens in longer text"""
        counter = TokenCounter("gpt-4")
        
        text = "The quick brown fox jumps over the lazy dog"
        token_count = counter.count_tokens(text)
        
        # Should be roughly 9-10 tokens
        assert 8 <= token_count <= 12
    
    def test_count_message_tokens(self):
        """Test counting tokens in messages"""
        counter = TokenCounter("gpt-4")
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "What is Python?"}
        ]
        
        token_count = counter.count_message_tokens(messages)
        
        # Should be more than just the text tokens due to formatting
        assert token_count > 0
        assert token_count > counter.count_tokens("You are a helpful assistantWhat is Python?")
    
    def test_count_message_tokens_empty(self):
        """Test counting tokens in empty message list"""
        counter = TokenCounter("gpt-4")
        
        assert counter.count_message_tokens([]) == 0
    
    def test_count_message_tokens_single_message(self):
        """Test counting tokens in single message"""
        counter = TokenCounter("gpt-4")
        
        messages = [{"role": "user", "content": "Hello"}]
        token_count = counter.count_message_tokens(messages)
        
        # Should include text + role + formatting overhead
        assert token_count > 1
    
    def test_calculate_cost_gpt4(self):
        """Test cost calculation for GPT-4"""
        counter = TokenCounter("gpt-4")
        
        # 1000 input tokens, 500 output tokens
        cost = counter.calculate_cost(1000, 500)
        
        # GPT-4 pricing: $0.03 per 1K input, $0.06 per 1K output
        # Expected: (1000/1000 * 0.03) + (500/1000 * 0.06) = 0.03 + 0.03 = 0.06
        expected = 0.06
        assert abs(cost - expected) < 0.001
    
    def test_calculate_cost_gpt35(self):
        """Test cost calculation for GPT-3.5"""
        counter = TokenCounter("gpt-3.5-turbo")
        
        # 1000 input tokens, 1000 output tokens
        cost = counter.calculate_cost(1000, 1000)
        
        # GPT-3.5 pricing: $0.0015 per 1K input, $0.002 per 1K output
        # Expected: (1000/1000 * 0.0015) + (1000/1000 * 0.002) = 0.0015 + 0.002 = 0.0035
        expected = 0.0035
        assert abs(cost - expected) < 0.0001
    
    def test_calculate_cost_zero_tokens(self):
        """Test cost calculation with zero tokens"""
        counter = TokenCounter("gpt-4")
        
        cost = counter.calculate_cost(0, 0)
        assert cost == 0.0
    
    def test_calculate_cost_only_input(self):
        """Test cost calculation with only input tokens"""
        counter = TokenCounter("gpt-4")
        
        cost = counter.calculate_cost(1000, 0)
        expected = 0.03  # 1000 tokens * $0.03 per 1K
        assert abs(cost - expected) < 0.001
    
    def test_calculate_cost_only_output(self):
        """Test cost calculation with only output tokens"""
        counter = TokenCounter("gpt-4")
        
        cost = counter.calculate_cost(0, 1000)
        expected = 0.06  # 1000 tokens * $0.06 per 1K
        assert abs(cost - expected) < 0.001
    
    def test_normalize_model_name_gpt4(self):
        """Test model name normalization for GPT-4 variants"""
        counter = TokenCounter("gpt-4-0613")
        model_key = counter._normalize_model_name("gpt-4-0613")
        assert model_key == "gpt-4"
        
        counter = TokenCounter("gpt-4-turbo")
        model_key = counter._normalize_model_name("gpt-4-turbo")
        assert model_key == "gpt-4-turbo"
    
    def test_normalize_model_name_claude(self):
        """Test model name normalization for Claude models"""
        counter = TokenCounter("claude-3-opus-20240229")
        model_key = counter._normalize_model_name("claude-3-opus-20240229")
        assert model_key == "claude-3-opus"
    
    def test_normalize_model_name_unknown(self):
        """Test model name normalization for unknown models"""
        counter = TokenCounter("unknown-model")
        model_key = counter._normalize_model_name("unknown-model")
        # Should default to gpt-3.5-turbo
        assert model_key == "gpt-3.5-turbo"
    
    def test_get_pricing_info_gpt4(self):
        """Test getting pricing info for GPT-4"""
        counter = TokenCounter("gpt-4")
        pricing = counter.get_pricing_info()
        
        assert 'input' in pricing
        assert 'output' in pricing
        assert pricing['input'] == 0.03
        assert pricing['output'] == 0.06
    
    def test_get_pricing_info_claude(self):
        """Test getting pricing info for Claude"""
        counter = TokenCounter("claude-3-sonnet")
        pricing = counter.get_pricing_info()
        
        assert 'input' in pricing
        assert 'output' in pricing
        assert pricing['input'] == 0.003
        assert pricing['output'] == 0.015
    
    def test_get_pricing_info_unknown(self):
        """Test getting pricing info for unknown model"""
        counter = TokenCounter("unknown-model-xyz")
        pricing = counter.get_pricing_info()
        
        # Should return default pricing
        assert 'input' in pricing
        assert 'output' in pricing
        assert pricing['input'] == 0.01
        assert pricing['output'] == 0.02
    
    def test_different_models(self):
        """Test that different models can be created"""
        gpt4 = TokenCounter("gpt-4")
        gpt35 = TokenCounter("gpt-3.5-turbo")
        claude = TokenCounter("claude-3-opus")
        
        text = "Hello world"
        
        # All should be able to count tokens
        assert gpt4.count_tokens(text) > 0
        assert gpt35.count_tokens(text) > 0
        assert claude.count_tokens(text) > 0
