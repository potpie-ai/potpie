![alt text](image.png)# Agent Evaluation Metrics and Token Tracking

This PR adds evaluation metrics and token tracking capabilities to enable better observability and cost management for the agent system.

## Motivation

The current implementation lacks visibility into agent performance and LLM costs:
- No metrics for agent response quality or execution time
- No tracking of LLM token usage and associated costs
- Limited ability to identify performance bottlenecks
- No data for optimization decisions

This PR addresses these gaps by providing a metrics collection framework and token tracking system.

## Features Added

### 1. Agent Evaluation Metrics (`app/modules/intelligence/agents/evaluation/`)

A comprehensive metrics collection system that tracks:
- **Timing metrics**: Response time, execution duration
- **Tool usage metrics**: Number of tools called, success/failure rates
- **Response metrics**: Length, word count
- **Quality metrics**: Extensible framework for completeness and relevance scores

**Key Components:**
- `AgentEvaluationMetrics`: Dataclass containing all metrics for a single interaction
- `MetricsCollector`: Helper class to track metrics throughout agent execution lifecycle

**Example Usage:**
```python
from app.modules.intelligence.agents.evaluation import MetricsCollector

collector = MetricsCollector()

# Start tracking
tracking_data = collector.start_tracking(
    agent_id="QnAAgent",
    conversation_id="conv_123",
    query="What is Python?"
)

# ... agent execution ...
collector.record_tool_call(tracking_data, "search_tool", success=True)

# Finalize
metrics = collector.finalize_metrics(tracking_data, response_text)

# Use metrics
print(f"Duration: {metrics.total_duration_ms}ms")
print(f"Tool success rate: {metrics.tool_success_rate}")
```

### 2. Token Tracking (`app/modules/intelligence/provider/token_counter.py`)

Accurate token counting and cost calculation for multiple LLM providers:
- Uses `tiktoken` for precise OpenAI token counting
- Supports GPT-4, GPT-3.5, Claude 3 (Opus, Sonnet, Haiku)
- Calculates costs based on current pricing
- Accounts for message formatting overhead

**Example Usage:**
```python
from app.modules.intelligence.provider.token_counter import TokenCounter

counter = TokenCounter("gpt-4")

# Count tokens
messages = [{"role": "user", "content": "What is Python?"}]
input_tokens = counter.count_message_tokens(messages)
output_tokens = counter.count_tokens(response)

# Calculate cost
cost = counter.calculate_cost(input_tokens, output_tokens)
print(f"Cost: ${cost:.4f}")
```

## 📁 Files Added

```text
app/modules/intelligence/agents/evaluation/
├── __init__.py
└── metrics.py                          # Metrics collection system

app/modules/intelligence/provider/
└── token_counter.py                    # Token counting and cost calculation

tests/intelligence/agents/
└── test_metrics.py                     # Comprehensive metrics tests (95%+ coverage)

tests/intelligence/provider/
└── test_token_counter.py               # Token counter tests (95%+ coverage)
```

## Testing

All new functionality includes comprehensive test coverage:

```bash
# Run metrics tests
pytest tests/intelligence/agents/test_metrics.py -v

# Run token counter tests
pytest tests/intelligence/provider/test_token_counter.py -v

# Run all new tests
pytest tests/intelligence/ -v
```

**Test Coverage:**
- 23 test cases for MetricsCollector
- 19 test cases for TokenCounter
- Edge cases covered: empty inputs, no tool calls, zero tokens
- Multiple model support validated
- Cost calculation accuracy verified

## Integration Path

These modules are designed to be easily integrated into existing agents:

### For Metrics:
```python
# In PydanticRagAgent or any agent base class:
from app.modules.intelligence.agents.evaluation import MetricsCollector

class MyAgent:
    def __init__(self, ...):
        self.metrics_collector = MetricsCollector()
    
    async def run(self, ctx):
        tracking = self.metrics_collector.start_tracking(...)
        # ... execution ...
        metrics = self.metrics_collector.finalize_metrics(tracking, response)
        logger.info("Agent metrics", **metrics.to_dict())
```

### For Token Tracking:
```python
# In ProviderService:
from app.modules.intelligence.provider.token_counter import TokenCounter

class ProviderService:
    def __init__(self, ...):
        self.token_counter = TokenCounter(self.model)
    
    async def call_llm(self, messages):
        input_tokens = self.token_counter.count_message_tokens(messages)
        response = await llm_call(messages)
        output_tokens = self.token_counter.count_tokens(response)
        cost = self.token_counter.calculate_cost(input_tokens, output_tokens)
        logger.info("LLM usage", input_tokens=input_tokens, cost=cost)
```

## Expected Benefits

- **Cost Visibility**: Real-time tracking of LLM costs to identify expensive operations
- **Performance Monitoring**: Measurement of agent response times and tool efficiency
- **Quality Metrics**: Foundation for automated agent evaluation
- **Optimization**: Data-driven decision making for agent improvements
- **Debugging**: Better insights for troubleshooting

## Design Decisions

- **Non-invasive**: Modules don't modify existing code, easy to integrate gradually
- **Extensible**: Easy to add new metrics or supported models
- **Type-safe**: Full type hints for better IDE support
- **Well-documented**: Comprehensive docstrings and examples
- **Tested**: High test coverage ensures reliability

## Future Enhancements

These modules provide the foundation for:
- Integration with observability platforms (LangSmith, Phoenix)
- Automated agent evaluation pipelines
- A/B testing different agent configurations
- Real-time cost alerting
- Quality score calculation using LLM-as-judge

## Checklist

- [x] Code follows project style guidelines
- [x] Comprehensive tests added (42 test cases total)
- [x] All tests pass
- [x] Documentation and docstrings included
- [x] No breaking changes to existing code
- [x] Type hints throughout
- [x] Ready for integration into existing agents

---

**Note**: This PR adds new modules without modifying existing code. Integration into existing agents can be done incrementally in follow-up PRs.
