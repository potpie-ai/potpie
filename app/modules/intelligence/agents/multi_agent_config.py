"""
Configuration for multi-agent system.

This module provides configuration options for enabling/disabling
multi-agent mode across the system.
"""

import os
from typing import Dict, Any, Optional


class MultiAgentConfig:
    """Configuration class for multi-agent system settings"""

    # Global multi-agent mode setting (default disabled — PydanticDeepAgent is the default runtime)
    ENABLE_MULTI_AGENT = os.getenv("ENABLE_MULTI_AGENT", "false").lower() == "true"

    # Per-agent type multi-agent settings (default disabled — opt in via env var)
    AGENT_MULTI_AGENT_SETTINGS = {
        "general_purpose_agent": os.getenv(
            "GENERAL_PURPOSE_MULTI_AGENT", "false"
        ).lower()
        == "true",
        "code_generation_agent": os.getenv("CODE_GEN_MULTI_AGENT", "false").lower()
        == "true",
        "codebase_qna_agent": os.getenv("QNA_MULTI_AGENT", "false").lower() == "true",
        "debugging_agent": os.getenv("DEBUG_MULTI_AGENT", "false").lower() == "true",
        "unit_test_agent": os.getenv("UNIT_TEST_MULTI_AGENT", "false").lower() == "true",
        "integration_test_agent": os.getenv(
            "INTEGRATION_TEST_MULTI_AGENT", "false"
        ).lower()
        == "true",
        "LLD_agent": os.getenv("LLD_MULTI_AGENT", "false").lower() == "true",
        "code_changes_agent": os.getenv("CODE_CHANGES_MULTI_AGENT", "false").lower()
        == "true",
        "sweb_debug_agent": os.getenv("SWEB_DEBUG_MULTI_AGENT", "false").lower()
        == "true",
    }

    # Custom agent multi-agent setting (default disabled)
    CUSTOM_AGENT_MULTI_AGENT = (
        os.getenv("CUSTOM_AGENT_MULTI_AGENT", "false").lower() == "true"
    )

    @classmethod
    def should_use_multi_agent(cls, agent_type: Optional[str] = None) -> bool:
        """
        Determine if multi-agent mode should be used for a specific agent type.

        Args:
            agent_type: The type of agent to check. If None, returns global setting.

        Returns:
            bool: True if multi-agent mode should be enabled (DEFAULT: False — PydanticDeepAgent is the default)
        """
        if not cls.ENABLE_MULTI_AGENT:
            return False

        if agent_type is None:
            return cls.ENABLE_MULTI_AGENT

        # Default to False for unknown agent types so PydanticDeepAgent stays the default.
        # Opt into multi-agent per agent via env var.
        return cls.AGENT_MULTI_AGENT_SETTINGS.get(agent_type, False)

    @classmethod
    def get_agent_config(cls, agent_type: str) -> Dict[str, Any]:
        """
        Get configuration for a specific agent type.

        Args:
            agent_type: The type of agent

        Returns:
            Dict containing agent configuration
        """
        return {
            "use_multi_agent": cls.should_use_multi_agent(agent_type),
            "agent_type": agent_type,
        }

    @classmethod
    def get_all_configs(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get all agent configurations.

        Returns:
            Dict mapping agent types to their configurations
        """
        configs = {}
        for agent_type in cls.AGENT_MULTI_AGENT_SETTINGS.keys():
            configs[agent_type] = cls.get_agent_config(agent_type)
        return configs


# Environment variable examples for configuration:
"""
# Global multi-agent mode (default: false — PydanticDeepAgent runs by default)
ENABLE_MULTI_AGENT=false

# Per-agent multi-agent opt-in (default: false for all)
GENERAL_PURPOSE_MULTI_AGENT=false
CODE_GEN_MULTI_AGENT=false
QNA_MULTI_AGENT=false
DEBUG_MULTI_AGENT=false
UNIT_TEST_MULTI_AGENT=false
INTEGRATION_TEST_MULTI_AGENT=false
LLD_MULTI_AGENT=false
CODE_CHANGES_MULTI_AGENT=false

# Custom agent multi-agent setting (default: false)
CUSTOM_AGENT_MULTI_AGENT=false

# To opt a specific agent into multi-agent mode, set both the global and per-agent flag to true:
# ENABLE_MULTI_AGENT=true
# QNA_MULTI_AGENT=true
"""
