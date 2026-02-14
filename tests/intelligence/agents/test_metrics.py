"""
Tests for Agent Evaluation Metrics Module
"""

import pytest
from datetime import datetime
from app.modules.intelligence.agents.evaluation.metrics import (
    MetricsCollector,
    AgentEvaluationMetrics
)


class TestAgentEvaluationMetrics:
    """Test cases for AgentEvaluationMetrics dataclass"""
    
    def test_metrics_creation(self):
        """Test creating metrics object"""
        start = datetime.now()
        end = datetime.now()
        
        metrics = AgentEvaluationMetrics(
            start_time=start,
            end_time=end,
            total_duration_ms=100.0,
            response_length=50,
            response_word_count=10,
            tools_called=["tool1", "tool2"],
            tool_call_count=2,
            tool_success_count=2,
            tool_failure_count=0,
            agent_id="test_agent",
            conversation_id="conv_123",
            query="test query"
        )
        
        assert metrics.agent_id == "test_agent"
        assert metrics.tool_call_count == 2
        assert metrics.tool_success_count == 2
    
    def test_tool_success_rate_all_success(self):
        """Test success rate with all tools succeeding"""
        start = datetime.now()
        end = datetime.now()
        
        metrics = AgentEvaluationMetrics(
            start_time=start,
            end_time=end,
            total_duration_ms=100.0,
            response_length=50,
            response_word_count=10,
            tools_called=["tool1", "tool2"],
            tool_call_count=2,
            tool_success_count=2,
            tool_failure_count=0,
            agent_id="test_agent",
            conversation_id="conv_123",
            query="test query"
        )
        
        assert metrics.tool_success_rate == 1.0
    
    def test_tool_success_rate_mixed(self):
        """Test success rate with mixed success/failure"""
        start = datetime.now()
        end = datetime.now()
        
        metrics = AgentEvaluationMetrics(
            start_time=start,
            end_time=end,
            total_duration_ms=100.0,
            response_length=50,
            response_word_count=10,
            tools_called=["tool1", "tool2", "tool3"],
            tool_call_count=3,
            tool_success_count=2,
            tool_failure_count=1,
            agent_id="test_agent",
            conversation_id="conv_123",
            query="test query"
        )
        
        assert abs(metrics.tool_success_rate - 2/3) < 0.001
    
    def test_tool_success_rate_no_tools(self):
        """Test success rate when no tools called"""
        start = datetime.now()
        end = datetime.now()
        
        metrics = AgentEvaluationMetrics(
            start_time=start,
            end_time=end,
            total_duration_ms=100.0,
            response_length=50,
            response_word_count=10,
            tools_called=[],
            tool_call_count=0,
            tool_success_count=0,
            tool_failure_count=0,
            agent_id="test_agent",
            conversation_id="conv_123",
            query="test query"
        )
        
        # No tools called should return 1.0 (perfect success, no failures)
        assert metrics.tool_success_rate == 1.0
    
    def test_to_dict(self):
        """Test converting metrics to dictionary"""
        start = datetime.now()
        end = datetime.now()
        
        metrics = AgentEvaluationMetrics(
            start_time=start,
            end_time=end,
            total_duration_ms=100.0,
            response_length=50,
            response_word_count=10,
            tools_called=["tool1"],
            tool_call_count=1,
            tool_success_count=1,
            tool_failure_count=0,
            agent_id="test_agent",
            conversation_id="conv_123",
            query="test query"
        )
        
        result = metrics.to_dict()
        
        assert isinstance(result, dict)
        assert result['agent_id'] == "test_agent"
        assert result['tool_call_count'] == 1
        assert isinstance(result['start_time'], str)  # Should be ISO format


class TestMetricsCollector:
    """Test cases for MetricsCollector class"""
    
    def test_start_tracking(self):
        """Test starting metric tracking"""
        collector = MetricsCollector()
        
        tracking_data = collector.start_tracking(
            agent_id="test_agent",
            conversation_id="conv_123",
            query="Test query"
        )
        
        assert tracking_data['agent_id'] == "test_agent"
        assert tracking_data['conversation_id'] == "conv_123"
        assert tracking_data['query'] == "Test query"
        assert 'start_time' in tracking_data
        assert isinstance(tracking_data['start_time'], datetime)
        assert isinstance(tracking_data['tools_called'], list)
        assert len(tracking_data['tools_called']) == 0
        assert tracking_data['tool_successes'] == 0
        assert tracking_data['tool_failures'] == 0
    
    def test_record_tool_call_success(self):
        """Test recording successful tool calls"""
        collector = MetricsCollector()
        tracking_data = collector.start_tracking("agent", "conv", "query")
        
        # Record successful tool call
        collector.record_tool_call(tracking_data, "search_tool", success=True)
        
        assert len(tracking_data['tools_called']) == 1
        assert tracking_data['tools_called'][0] == "search_tool"
        assert tracking_data['tool_successes'] == 1
        assert tracking_data['tool_failures'] == 0
    
    def test_record_tool_call_failure(self):
        """Test recording failed tool calls"""
        collector = MetricsCollector()
        tracking_data = collector.start_tracking("agent", "conv", "query")
        
        # Record failed tool call
        collector.record_tool_call(tracking_data, "analyze_tool", success=False)
        
        assert len(tracking_data['tools_called']) == 1
        assert tracking_data['tools_called'][0] == "analyze_tool"
        assert tracking_data['tool_successes'] == 0
        assert tracking_data['tool_failures'] == 1
    
    def test_record_multiple_tool_calls(self):
        """Test recording multiple tool calls"""
        collector = MetricsCollector()
        tracking_data = collector.start_tracking("agent", "conv", "query")
        
        # Record multiple tool calls
        collector.record_tool_call(tracking_data, "tool1", success=True)
        collector.record_tool_call(tracking_data, "tool2", success=True)
        collector.record_tool_call(tracking_data, "tool3", success=False)
        
        assert len(tracking_data['tools_called']) == 3
        assert tracking_data['tool_successes'] == 2
        assert tracking_data['tool_failures'] == 1
    
    def test_finalize_metrics(self):
        """Test finalizing metrics"""
        collector = MetricsCollector()
        tracking_data = collector.start_tracking("agent", "conv", "Test query")
        
        # Simulate some tool calls
        collector.record_tool_call(tracking_data, "tool1", success=True)
        collector.record_tool_call(tracking_data, "tool2", success=True)
        
        # Finalize with response
        response = "This is a test response with multiple words"
        metrics = collector.finalize_metrics(tracking_data, response)
        
        assert isinstance(metrics, AgentEvaluationMetrics)
        assert metrics.agent_id == "agent"
        assert metrics.conversation_id == "conv"
        assert metrics.query == "Test query"
        assert metrics.tool_call_count == 2
        assert metrics.tool_success_count == 2
        assert metrics.tool_success_rate == 1.0
        assert metrics.response_length == len(response)
        assert metrics.response_word_count == 8
        assert metrics.total_duration_ms > 0
        assert len(collector.metrics) == 1
    
    def test_finalize_metrics_no_tools(self):
        """Test finalizing metrics when no tools were called"""
        collector = MetricsCollector()
        tracking_data = collector.start_tracking("agent", "conv", "query")
        
        # Finalize without any tool calls
        response = "Simple response"
        metrics = collector.finalize_metrics(tracking_data, response)
        
        assert metrics.tool_call_count == 0
        assert metrics.tool_success_rate == 1.0  # No failures = perfect
        assert metrics.response_length == len(response)
    
    def test_multiple_metrics_collection(self):
        """Test collecting multiple metrics"""
        collector = MetricsCollector()
        
        # First interaction
        tracking1 = collector.start_tracking("agent1", "conv1", "query1")
        metrics1 = collector.finalize_metrics(tracking1, "response1")
        
        # Second interaction
        tracking2 = collector.start_tracking("agent2", "conv2", "query2")
        metrics2 = collector.finalize_metrics(tracking2, "response2")
        
        assert len(collector.metrics) == 2
        assert collector.metrics[0].agent_id == "agent1"
        assert collector.metrics[1].agent_id == "agent2"
    
    def test_get_all_metrics(self):
        """Test retrieving all collected metrics"""
        collector = MetricsCollector()
        
        # Collect some metrics
        for i in range(3):
            tracking = collector.start_tracking(f"agent{i}", f"conv{i}", f"query{i}")
            collector.finalize_metrics(tracking, f"response{i}")
        
        all_metrics = collector.get_all_metrics()
        
        assert len(all_metrics) == 3
        assert all(isinstance(m, AgentEvaluationMetrics) for m in all_metrics)
    
    def test_clear_metrics(self):
        """Test clearing stored metrics"""
        collector = MetricsCollector()
        
        # Collect some metrics
        tracking = collector.start_tracking("agent", "conv", "query")
        collector.finalize_metrics(tracking, "response")
        
        assert len(collector.metrics) == 1
        
        # Clear metrics
        collector.clear_metrics()
        
        assert len(collector.metrics) == 0
