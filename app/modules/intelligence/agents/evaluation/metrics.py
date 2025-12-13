"""
Agent Evaluation Metrics Module

This module provides metrics collection and tracking for LLM agent interactions.
It enables observability by tracking response times, tool usage, and quality metrics.
"""

import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict, field
from datetime import datetime


@dataclass
class AgentEvaluationMetrics:
    """
    Metrics for a single agent interaction.
    
    This class captures comprehensive metrics about an agent's execution,
    including timing, tool usage, and response characteristics.
    """
    
    # Timing metrics
    start_time: datetime
    end_time: datetime
    total_duration_ms: float
    
    # Response metrics
    response_length: int
    response_word_count: int
    
    # Tool usage metrics
    tools_called: List[str]
    tool_call_count: int
    tool_success_count: int
    tool_failure_count: int
    
    # Context
    agent_id: str
    conversation_id: str
    query: str
    
    # Quality metrics (0-1 scale) - can be extended later
    completeness_score: float = 0.0
    relevance_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for logging/storage"""
        data = asdict(self)
        # Convert datetime objects to ISO format strings
        data['start_time'] = self.start_time.isoformat()
        data['end_time'] = self.end_time.isoformat()
        return data
    
    @property
    def tool_success_rate(self) -> float:
        """
        Calculate the success rate of tool calls.
        
        Returns:
            float: Success rate between 0.0 and 1.0
                   Returns 1.0 if no tools were called (no failures)
        """
        if self.tool_call_count == 0:
            return 1.0  # No tools called = no failures
        return self.tool_success_count / self.tool_call_count


class MetricsCollector:
    """
    Collects and manages metrics during agent execution.
    
    This class provides a simple interface to track agent performance
    throughout the execution lifecycle.
    
    Example usage:
        collector = MetricsCollector()
        tracking_data = collector.start_tracking("my_agent", "conv_123", "What is Python?")
        
        # ... agent execution ...
        collector.record_tool_call(tracking_data, "search_tool", success=True)
        
        # ... more execution ...
        metrics = collector.finalize_metrics(tracking_data, response_text)
        
        # Log or store metrics
        logger.info("Agent metrics", **metrics.to_dict())
    """
    
    def __init__(self):
        self.metrics: List[AgentEvaluationMetrics] = []
    
    def start_tracking(
        self, 
        agent_id: str, 
        conversation_id: str, 
        query: str
    ) -> Dict[str, Any]:
        """
        Start tracking a new agent interaction.
        
        Args:
            agent_id: Identifier for the agent (e.g., "QnAAgent", "DebugAgent")
            conversation_id: ID of the conversation/session
            query: The user's query or input
            
        Returns:
            Dict containing tracking state to be passed to other methods
        """
        return {
            'agent_id': agent_id,
            'conversation_id': conversation_id,
            'query': query,
            'start_time': datetime.now(),
            'tools_called': [],
            'tool_successes': 0,
            'tool_failures': 0
        }
    
    def record_tool_call(
        self, 
        tracking_data: Dict[str, Any], 
        tool_name: str, 
        success: bool
    ) -> None:
        """
        Record a tool call during agent execution.
        
        Args:
            tracking_data: The tracking state from start_tracking()
            tool_name: Name of the tool that was called
            success: Whether the tool call succeeded
        """
        tracking_data['tools_called'].append(tool_name)
        if success:
            tracking_data['tool_successes'] += 1
        else:
            tracking_data['tool_failures'] += 1
    
    def finalize_metrics(
        self, 
        tracking_data: Dict[str, Any], 
        response: str
    ) -> AgentEvaluationMetrics:
        """
        Finalize and store metrics after agent execution completes.
        
        Args:
            tracking_data: The tracking state from start_tracking()
            response: The final response text from the agent
            
        Returns:
            AgentEvaluationMetrics object containing all collected metrics
        """
        end_time = datetime.now()
        duration = (end_time - tracking_data['start_time']).total_seconds() * 1000
        
        metrics = AgentEvaluationMetrics(
            start_time=tracking_data['start_time'],
            end_time=end_time,
            total_duration_ms=duration,
            response_length=len(response),
            response_word_count=len(response.split()),
            tools_called=tracking_data['tools_called'],
            tool_call_count=len(tracking_data['tools_called']),
            tool_success_count=tracking_data['tool_successes'],
            tool_failure_count=tracking_data['tool_failures'],
            agent_id=tracking_data['agent_id'],
            conversation_id=tracking_data['conversation_id'],
            query=tracking_data['query']
        )
        
        self.metrics.append(metrics)
        return metrics
    
    def get_all_metrics(self) -> List[AgentEvaluationMetrics]:
        """
        Get all collected metrics.
        
        Returns:
            List of all AgentEvaluationMetrics objects collected so far
        """
        return self.metrics
    
    def clear_metrics(self) -> None:
        """Clear all stored metrics"""
        self.metrics = []
