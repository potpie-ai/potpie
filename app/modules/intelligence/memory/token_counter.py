"""
Token counting service for accurate input token measurement across providers.
Supports OpenAI (tiktoken), Anthropic (native API), and HuggingFace models.

This service is dynamic and reusable - it accepts model names and thresholds,
looks up context windows internally, and returns threshold breach status.
"""

import logging
import os
from typing import List, Optional, Tuple, Union, Any
from enum import Enum

# Disable tokenizer parallelism to avoid issues with forked processes (e.g., Celery workers)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)


class TokenCounterProvider(str, Enum):
    """Supported token counter providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OPENROUTER = "openrouter"


class TokenCounterService:
    """
    Dynamic token counting service that works with any model.
    
    Features:
    - Provider-agnostic: Routes to correct tokenizer based on model
    - Self-contained: Looks up context windows and capabilities internally
    - Threshold checking: Returns whether threshold is exceeded
    - Caching: Tokenizers and token counts cached for performance
    """
    
    def __init__(self, llm_provider):
        """
        Initialize with provider service to detect model type and get API keys.
        
        Args:
            llm_provider: ProviderService instance
        """
        self.llm_provider = llm_provider
        
        # Lazy-loaded tokenizers (loaded on first use)
        self._tiktoken_cache = {}
        self._hf_tokenizer_cache = {}
        self._anthropic_client = None
        
        # Token count cache to avoid repeated Anthropic API calls
        self._anthropic_token_cache = {}
        
        # Track last logged milestone for progress logging
        self._last_logged_milestone = None
        
    def check_token_threshold(
        self,
        messages: List[Union[Any, dict, str]],
        system_instructions: Optional[str] = None,
        tools: Optional[List[Union[Any, dict]]] = None,
        model_name: Optional[str] = None,
        threshold_percentage: float = 0.90,
        config_type: str = "chat",
        context_window: Optional[int] = None,
        token_threshold: Optional[int] = None
    ) -> Tuple[bool, int, int]:
        """
        Check if token usage exceeds threshold for a given model.
        
        This is the main entry point for agents. It:
        1. Gets the model's context window
        2. Counts all tokens (system + tools + messages)
        3. Calculates threshold
        4. Returns whether threshold is exceeded
        
        Args:
            messages: List of ModelMessage objects from pydantic-ai
            system_instructions: System prompt/instructions text (optional)
            tools: List of tool definitions/schemas (optional)
            model_name: Model identifier (e.g., "openai/gpt-4o"). If None, uses llm_provider config
            threshold_percentage: Percentage of context window to trigger (default: 0.90 = 90%)
            config_type: "chat" or "inference" for selecting model config if model_name not provided
            context_window: Optional context window size (overrides lookup if provided)
            token_threshold: Optional token threshold (overrides calculation if provided)
            
        Returns:
            Tuple of (should_compress: bool, total_tokens: int, context_window: int)
        """
        try:
            # Use provided context_window and token_threshold if available, otherwise calculate
            if context_window is None or token_threshold is None:
                # Determine model name if not provided
                if model_name is None:
                    config = (
                        self.llm_provider.chat_config 
                        if config_type == "chat" 
                        else self.llm_provider.inference_config
                    )
                    model_name = config.model
                
                # Get context window for this model if not provided
                if context_window is None:
                    from app.modules.intelligence.provider.llm_config import get_context_window_for_model
                    context_window = get_context_window_for_model(model_name)
                
                # Calculate threshold if not provided
                if token_threshold is None:
                    token_threshold = int(context_window * threshold_percentage)
            
            # Count tokens
            token_count = self.count_tokens(
                messages=messages,
                system_instructions=system_instructions,
                tools=tools,
                model_name=model_name,
                config_type=config_type
            )
            
            # Log progress at threshold percentages (every 10%)
            if token_count > 0:
                usage_pct = (token_count / context_window) * 100
                milestone = int(usage_pct // 10) * 10
                if self._last_logged_milestone is None:
                    # First time, always log
                    logger.info(
                        f"📊 Input Tokens: {token_count:,}/{context_window:,} ({usage_pct:.1f}%)"
                    )
                    self._last_logged_milestone = milestone
                elif milestone != self._last_logged_milestone and milestone % 10 == 0:
                    # Log when crossing a new 10% milestone
                    logger.info(
                        f"📊 Input Tokens: {token_count:,}/{context_window:,} ({usage_pct:.1f}%)"
                    )
                    self._last_logged_milestone = milestone
            
            # Check if threshold exceeded
            exceeds_threshold = token_count > token_threshold
            
            if exceeds_threshold:
                logger.warning(
                    f"⚠️  Threshold exceeded: {token_count:,} > {token_threshold:,} "
                    f"({threshold_percentage*100:.0f}% of {context_window:,})"
                )
            
            return (exceeds_threshold, token_count, context_window)
            
        except Exception as e:
            # Capture context information safely for error logging
            model_info = "model=unknown"
            msg_count = 0
            
            try:
                if 'model_name' in locals() and model_name:
                    model_info = f"model={model_name}"
            except Exception:
                pass
            
            try:
                msg_count = len(messages) if messages else 0
            except Exception:
                pass
            
            logger.error(
                f"Threshold check failed: {e}. "
                f"Context: {model_info}, messages_count={msg_count}, "
                f"context_window={'provided' if context_window else 'calculated'}, "
                f"token_threshold={'provided' if token_threshold else 'calculated'}",
                exc_info=True
            )
            # On error, don't trigger compression - continue execution
            # Return explicit error indication: (should_compress=False, token_count=0, context_window=0)
            return (False, 0, 0)
    
    def count_tokens(
        self,
        messages: List[Union[Any, dict, str]],
        system_instructions: Optional[str] = None,
        tools: Optional[List[Union[Any, dict]]] = None,
        model_name: Optional[str] = None,
        config_type: str = "chat"
    ) -> int:
        """
        Count tokens in a complete agent context including messages, system, and tools.
        
        Args:
            messages: List of ModelMessage objects from pydantic-ai
            system_instructions: System prompt/instructions text (optional)
            tools: List of tool definitions/schemas (optional)
            model_name: Optional model override (uses llm_provider config if None)
            config_type: "chat" or "inference" for selecting model config
            
        Returns:
            Token count as integer, or 0 if counting fails
        """
        try:
            # Determine model and provider
            if model_name is None:
                config = (
                    self.llm_provider.chat_config 
                    if config_type == "chat" 
                    else self.llm_provider.inference_config
                )
                model_name = config.model
                provider = config.provider
            else:
                from app.modules.intelligence.provider.llm_config import get_config_for_model
                model_config = get_config_for_model(model_name)
                provider = model_config.get("provider", "openai")
            
            logger.info(f"💭 Counting tokens for model: {model_name}, provider: {provider}")
            logger.info(f"   Input: {len(messages)} messages, system={'yes' if system_instructions else 'no'}, tools={len(tools) if tools else 0}")
            
            # Convert pydantic-ai messages to dict format
            message_dicts = self._convert_messages_to_dicts(messages)
            logger.info(f"   Converted to {len(message_dicts)} message dicts")
            
            # Route to appropriate counter
            if provider == "openai":
                return self._count_openai_tokens(
                    message_dicts, 
                    model_name, 
                    system_instructions=system_instructions,
                    tools=tools
                )
            elif provider == "anthropic":
                return self._count_anthropic_tokens(
                    message_dicts, 
                    model_name,
                    system_instructions=system_instructions,
                    tools=tools
                )
            elif provider == "gemini":
                # Gemini models don't have public tokenizers - use fallback estimation
                logger.info(f"Using fallback estimation for Gemini model: {model_name}")
                return self._fallback_estimate_complete(message_dicts, system_instructions, tools)
            else:
                # HuggingFace for everything else (deepseek, llama via openrouter)
                return self._count_hf_tokens(
                    message_dicts, 
                    model_name,
                    system_instructions=system_instructions,
                    tools=tools
                )
                
        except Exception as e:
            logger.error(f"Token counting failed: {e}", exc_info=True)
            # Fallback: estimate based on character count (rough: 1 token ≈ 4 chars)
            return self._fallback_estimate(messages)
    
    def _convert_messages_to_dicts(self, messages: List[Union[Any, dict, str]]) -> List[dict]:
        """
        Convert pydantic-ai ModelMessage objects to simple dicts.
        
        Format: [{"role": "user", "content": "..."}, ...]
        
        Handles:
        - ModelResponse objects (with parts including tool calls and results)
        - String messages
        - Pre-converted dict messages
        
        CRITICAL: This properly extracts tool results which can contain HUGE amounts
        of data (file contents, graph data, etc.) that fill up the context window.
        """
        result = []
        
        # Debug: Log incoming message structure
        logger.info(f"🔬 Converting {len(messages)} messages to dicts")
        for idx, msg in enumerate(messages):
            msg_type = type(msg).__name__
            if hasattr(msg, 'parts'):
                part_types = [type(p).__name__ for p in msg.parts]
                logger.info(f"  Message {idx}: {msg_type} with {len(msg.parts)} parts: {part_types}")
        
        for msg in messages:
            # Handle ModelResponse with parts (most common case)
            if hasattr(msg, 'parts'):
                content_parts = []
                
                for part in msg.parts:
                    # Check what type of part this is and handle accordingly
                    part_type = type(part).__name__
                    has_tool_name = hasattr(part, 'tool_name')
                    has_args = hasattr(part, 'args')
                    has_content = hasattr(part, 'content')
                    
                    logger.debug(f"    Part: {part_type}, tool_name={has_tool_name}, args={has_args}, content={has_content}")
                    
                    # Handle tool calls (ToolCallPart) - has tool_name and args
                    if hasattr(part, 'tool_name') and hasattr(part, 'args'):
                        # Tool calls have name and args - serialize them
                        import json
                        try:
                            args_dict = part.args if isinstance(part.args, dict) else {}
                            tool_call_str = f"[Tool Call: {part.tool_name}({json.dumps(args_dict)})]"
                            content_parts.append(tool_call_str)
                            logger.debug(f"      -> Added tool call: {part.tool_name}")
                        except Exception:
                            content_parts.append(f"[Tool Call: {part.tool_name}]")
                    
                    # Handle tool results (ToolReturnPart) - THIS IS CRITICAL
                    # Tool results contain file contents, graph data, etc.
                    # These have tool_name but NOT args (distinguishes from tool calls)
                    elif hasattr(part, 'tool_name') and not hasattr(part, 'args'):
                        # This is a tool result - the content can be MASSIVE (files, graph data, etc.)
                        if hasattr(part, 'content'):
                            tool_result = str(part.content)
                            # Don't add prefix/suffix, just count the raw content
                            content_parts.append(tool_result)
                            logger.info(f"🔍 Counting tool result from '{part.tool_name}': {len(tool_result):,} chars (~{len(tool_result)//4:,} tokens)")
                        else:
                            logger.warning(f"      -> Tool result '{part.tool_name}' has no content!")
                    
                    # Handle regular text content (TextPart)
                    elif hasattr(part, 'content'):
                        content_parts.append(str(part.content))
                        logger.debug(f"      -> Added text content: {len(str(part.content))} chars")
                
                content = " ".join(content_parts)
                
                # Determine role
                role = getattr(msg, 'role', 'assistant')
                
                if content:  # Only add if there's actual content
                    result.append({"role": role, "content": content})
            
            # Handle string messages (from history)
            elif isinstance(msg, str):
                result.append({"role": "user", "content": msg})
            
            # Handle pre-converted dict messages
            elif isinstance(msg, dict):
                result.append(msg)
            
            else:
                logger.warning(f"Unknown message type: {type(msg)}")
                
        return result
    
    def _count_openai_tokens(
        self, 
        messages: List[dict], 
        model: str,
        system_instructions: Optional[str] = None,
        tools: Optional[List] = None
    ) -> int:
        """
        Count tokens using tiktoken for OpenAI models.
        Includes system instructions, tool schemas, and messages.
        
        References:
        - https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken
        - https://platform.openai.com/docs/guides/function-calling
        """
        try:
            import tiktoken
            import json
            
            # Extract model name from full identifier (e.g., "openai/gpt-4o" -> "gpt-4o")
            model_name = model.split("/")[-1] if "/" in model else model
            
            # Get or create tokenizer
            if model_name not in self._tiktoken_cache:
                try:
                    self._tiktoken_cache[model_name] = tiktoken.encoding_for_model(model_name)
                except KeyError:
                    # Fallback to cl100k_base for unknown models
                    logger.warning(f"Model {model_name} not found in tiktoken, using cl100k_base")
                    self._tiktoken_cache[model_name] = tiktoken.get_encoding("cl100k_base")
            
            enc = self._tiktoken_cache[model_name]
            
            total_tokens = 0
            
            # 1. Count system instruction tokens
            if system_instructions:
                # System message format: <|im_start|>system\n{content}<|im_end|>
                system_tokens = len(enc.encode(system_instructions))
                total_tokens += system_tokens
                total_tokens += 4  # Overhead for system message wrapper
                logger.info(f"  📋 System instructions: {system_tokens:,} tokens")
            
            # 2. Count tool schema tokens (if tools are provided)
            if tools and len(tools) > 0:
                tool_tokens = 0
                for tool in tools:
                    # Serialize tool definition to JSON
                    tool_json = json.dumps(self._extract_tool_schema(tool))
                    tool_tokens += len(enc.encode(tool_json))
                
                # Add overhead for tools wrapper (~10 tokens per tool)
                tool_tokens += len(tools) * 10
                total_tokens += tool_tokens
                logger.info(f"  🔧 Tool schemas ({len(tools)} tools): {tool_tokens:,} tokens")
            
            # 3. Count message history tokens
            message_tokens = 0
            for msg in messages:
                # Add tokens for role and content
                message_tokens += len(enc.encode(msg.get("role", "")))
                content = msg.get("content", "")
                
                # Handle tool calls in messages (they have special format)
                if isinstance(content, dict):
                    content = json.dumps(content)
                
                message_tokens += len(enc.encode(str(content)))
                # Add overhead per message (ChatML format: ~4 tokens per message)
                message_tokens += 4
            
            total_tokens += message_tokens
            logger.info(f"  💬 Message history ({len(messages)} messages): {message_tokens:,} tokens")
            
            # Add 3 tokens for assistant reply priming
            total_tokens += 3
            
            logger.info(
                f"✅ OpenAI TOTAL: {total_tokens:,} tokens "
                f"(system: {len(enc.encode(system_instructions or ''))}, "
                f"tools: {len(tools) if tools else 0}, "
                f"messages: {len(messages)})"
            )
            return total_tokens
            
        except Exception as e:
            logger.error(f"OpenAI token counting failed: {e}", exc_info=True)
            return self._fallback_estimate_complete(messages, system_instructions, tools)
    
    def _count_anthropic_tokens(
        self, 
        messages: List[dict], 
        model: str,
        system_instructions: Optional[str] = None,
        tools: Optional[List] = None
    ) -> int:
        """
        Count tokens using Anthropic's native counter.
        Includes system instructions, tool schemas, and messages.
        
        Uses caching to avoid repeated API calls for same message set.
        
        References:
        - https://docs.anthropic.com/en/api/messages-count-tokens
        """
        try:
            # Create cache key from message content
            import hashlib
            import json
            
            cache_key = hashlib.md5(
                json.dumps({
                    "messages": messages,
                    "system": system_instructions or "",
                    "tools": len(tools) if tools else 0,
                    "model": model
                }, sort_keys=True).encode()
            ).hexdigest()
            
            # Check cache first
            if cache_key in self._anthropic_token_cache:
                cached_count = self._anthropic_token_cache[cache_key]
                logger.debug(f"Using cached Anthropic token count: {cached_count}")
                return cached_count
            
            # Import anthropic client
            if self._anthropic_client is None:
                from anthropic import Anthropic
                import os
                
                # Get API key
                api_key = self.llm_provider._get_api_key("anthropic")
                if not api_key:
                    api_key = os.getenv("ANTHROPIC_API_KEY")
                
                if not api_key:
                    logger.warning("No Anthropic API key found, using fallback estimation")
                    return self._fallback_estimate_complete(messages, system_instructions, tools)
                
                self._anthropic_client = Anthropic(api_key=api_key)
            
            # Extract model name
            model_name = model.split("/")[-1] if "/" in model else model
            
            # Convert to Anthropic's message format
            anthropic_messages = []
            for msg in messages:
                role = "user" if msg["role"] == "user" else "assistant"
                anthropic_messages.append({
                    "role": role,
                    "content": msg["content"]
                })
            
            # Prepare count_tokens parameters
            count_params = {
                "model": model_name,
                "messages": anthropic_messages
            }
            
            # Add system instructions if provided
            if system_instructions:
                count_params["system"] = system_instructions
            
            # Add tools if provided (convert to Anthropic tool format)
            if tools and len(tools) > 0:
                anthropic_tools = []
                for tool in tools:
                    tool_schema = self._extract_tool_schema(tool)
                    anthropic_tools.append({
                        "name": tool_schema.get("name", "unknown"),
                        "description": tool_schema.get("description", ""),
                        "input_schema": tool_schema.get("parameters", {})
                    })
                count_params["tools"] = anthropic_tools
            
            # Call Anthropic's token counting API
            # This counts everything: system + tools + messages
            response = self._anthropic_client.messages.count_tokens(**count_params)
            
            token_count = response.input_tokens
            
            # Cache the result
            self._anthropic_token_cache[cache_key] = token_count
            
            logger.info(
                f"✅ Anthropic TOTAL: {token_count:,} tokens "
                f"(system: {'yes' if system_instructions else 'no'}, "
                f"tools: {len(tools) if tools else 0}, "
                f"messages: {len(messages)})"
            )
            return token_count
            
        except Exception as e:
            logger.error(f"Anthropic token counting failed: {e}", exc_info=True)
            return self._fallback_estimate_complete(messages, system_instructions, tools)
    
    def _count_hf_tokens(
        self, 
        messages: List[dict], 
        model: str,
        system_instructions: Optional[str] = None,
        tools: Optional[List] = None
    ) -> int:
        """
        Count tokens using HuggingFace transformers.
        Includes system instructions, tool schemas (if template supports), and messages.
        
        References:
        - https://huggingface.co/docs/transformers/en/chat_templating
        - https://huggingface.co/docs/transformers/v4.51.3/en/chat_extras (tools)
        """
        try:
            from transformers import AutoTokenizer
            import json
            
            # Extract model identifier for HF
            # For openrouter models, use base model name
            if "openrouter" in model:
                # e.g., "openrouter/deepseek/deepseek-chat-v3-0324" -> "deepseek/deepseek-chat-v3-0324"
                parts = model.split("/", 1)
                if len(parts) > 1:
                    model_id = parts[1]
                else:
                    model_id = model
            else:
                model_id = model
            
            # Get or create tokenizer
            if model_id not in self._hf_tokenizer_cache:
                logger.info(f"Loading HuggingFace tokenizer for: {model_id}")
                self._hf_tokenizer_cache[model_id] = AutoTokenizer.from_pretrained(
                    model_id,
                    trust_remote_code=True  # Required for some models
                )
            
            tokenizer = self._hf_tokenizer_cache[model_id]
            
            # Prepare messages with system instruction
            chat_messages = messages.copy()
            if system_instructions:
                # Add system message at the start
                chat_messages.insert(0, {
                    "role": "system",
                    "content": system_instructions
                })
            
            # Try to include tools in chat template (if supported)
            template_kwargs = {
                "tokenize": False,
                "add_generation_prompt": True
            }
            
            # Some models support tools parameter in chat template
            if tools and len(tools) > 0:
                try:
                    tool_schemas = [self._extract_tool_schema(tool) for tool in tools]
                    template_kwargs["tools"] = tool_schemas
                except Exception as e:
                    logger.debug(f"Model {model_id} may not support tools in template: {e}")
            
            # Apply chat template
            try:
                rendered = tokenizer.apply_chat_template(
                    chat_messages,
                    **template_kwargs
                )
            except Exception as e:
                # Fallback without tools if template doesn't support it
                logger.debug(f"Falling back to template without tools: {e}")
                rendered = tokenizer.apply_chat_template(
                    chat_messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                # Manually append tool schemas to estimate (rough)
                if tools and len(tools) > 0:
                    tool_json = json.dumps([self._extract_tool_schema(t) for t in tools])
                    rendered += f"\n\nTools: {tool_json}"
            
            token_count = len(tokenizer.encode(rendered, add_special_tokens=True))
            logger.info(
                f"✅ HuggingFace TOTAL: {token_count:,} tokens "
                f"(system: {'yes' if system_instructions else 'no'}, "
                f"tools: {len(tools) if tools else 0}, "
                f"messages: {len(messages)})"
            )
            return token_count
            
        except Exception as e:
            logger.error(f"HuggingFace token counting failed: {e}", exc_info=True)
            return self._fallback_estimate_complete(messages, system_instructions, tools)
    
    def _extract_tool_schema(self, tool) -> dict:
        """
        Extract tool schema from various tool formats.
        
        Handles:
        - LangChain StructuredTool objects (primary use case)
        - Pydantic-ai Tool objects
        - Raw dict schemas
        """
        try:
            # If it's already a dict, return it
            if isinstance(tool, dict):
                return tool
            
            # Handle LangChain StructuredTool (PRIMARY - this is what we use)
            if hasattr(tool, 'args_schema') and hasattr(tool, 'name'):
                schema = {
                    "name": tool.name,
                    "description": getattr(tool, 'description', ''),
                }
                
                # Extract JSON schema from Pydantic model
                if tool.args_schema and hasattr(tool.args_schema, 'model_json_schema'):
                    schema["parameters"] = tool.args_schema.model_json_schema()
                else:
                    schema["parameters"] = {"type": "object", "properties": {}}
                
                return schema
            
            # Handle pydantic-ai Tool
            if hasattr(tool, 'name') and hasattr(tool, 'description'):
                schema = {
                    "name": tool.name,
                    "description": tool.description or "",
                }
                
                # Try to extract parameters/function signature
                if hasattr(tool, 'function'):
                    import inspect
                    sig = inspect.signature(tool.function)
                    # Build basic parameter schema
                    params = {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                    for param_name, param in sig.parameters.items():
                        if param_name not in ['self', 'ctx']:
                            params["properties"][param_name] = {
                                "type": "string",  # Rough approximation
                                "description": ""
                            }
                            if param.default == inspect.Parameter.empty:
                                params["required"].append(param_name)
                    schema["parameters"] = params
                
                return schema
            
            # Fallback: return minimal schema
            return {
                "name": str(tool),
                "description": "",
                "parameters": {"type": "object", "properties": {}}
            }
            
        except Exception as e:
            logger.debug(f"Failed to extract tool schema: {e}")
            return {
                "name": "unknown",
                "description": "",
                "parameters": {"type": "object", "properties": {}}
            }
    
    def _fallback_estimate(self, messages) -> int:
        """
        Fallback estimation when proper counting fails (messages only).
        Uses rough heuristic: 1 token ≈ 4 characters.
        """
        try:
            total_chars = 0
            for msg in messages:
                if hasattr(msg, 'parts'):
                    for part in msg.parts:
                        if hasattr(part, 'content'):
                            total_chars += len(str(part.content))
                elif isinstance(msg, str):
                    total_chars += len(msg)
                elif isinstance(msg, dict):
                    total_chars += len(str(msg.get('content', '')))
            
            estimated_tokens = total_chars // 4
            logger.warning(f"Using fallback token estimation: ~{estimated_tokens} tokens")
            return estimated_tokens
            
        except Exception as e:
            logger.error(f"Even fallback estimation failed: {e}")
            return 0
    
    def _fallback_estimate_complete(
        self, 
        messages, 
        system_instructions: Optional[str] = None,
        tools: Optional[List] = None
    ) -> int:
        """
        Complete fallback estimation including system and tools.
        Uses rough heuristic: 1 token ≈ 4 characters.
        """
        try:
            total_chars = 0
            
            # Count system instructions
            if system_instructions:
                total_chars += len(system_instructions)
            
            # Count tool schemas (rough JSON serialization)
            if tools:
                import json
                for tool in tools:
                    tool_schema = self._extract_tool_schema(tool)
                    total_chars += len(json.dumps(tool_schema))
            
            # Count messages
            for msg in messages:
                if hasattr(msg, 'parts'):
                    for part in msg.parts:
                        if hasattr(part, 'content'):
                            total_chars += len(str(part.content))
                elif isinstance(msg, str):
                    total_chars += len(msg)
                elif isinstance(msg, dict):
                    total_chars += len(str(msg.get('content', '')))
            
            estimated_tokens = total_chars // 4
            logger.warning(
                f"Using complete fallback estimation: ~{estimated_tokens} tokens "
                f"(system: {'yes' if system_instructions else 'no'}, "
                f"tools: {len(tools) if tools else 0})"
            )
            return estimated_tokens
            
        except Exception as e:
            logger.error(f"Complete fallback estimation failed: {e}")
            return 0

