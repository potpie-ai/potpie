"""
SSO Provider Registry with Singleton Pattern

This module provides a registry for SSO providers using the singleton pattern.
Since SSO providers are stateless (they only read configuration from environment
variables), they can be safely shared across all requests.

Benefits:
- Reduced memory usage (one instance per provider type)
- Faster initialization (providers created once)
- Thread-safe access to shared provider instances
"""

import threading
from typing import Dict, Optional, Type
from app.modules.auth.sso_providers.base_provider import BaseSSOProvider
from app.modules.auth.sso_providers.google_provider import GoogleSSOProvider

logger = None  # Will be initialized when needed


class SSOProviderRegistry:
    """
    Registry for SSO providers using singleton pattern.
    
    Each provider type is instantiated once and reused across all requests.
    This is safe because providers are stateless - they only read from
    environment variables or configuration passed at initialization.
    """
    
    _instances: Dict[str, BaseSSOProvider] = {}
    _lock = threading.Lock()
    _providers: Dict[str, Type[BaseSSOProvider]] = {
        "google": GoogleSSOProvider,
    }
    
    @classmethod
    def get_provider(cls, provider_name: str, config: Optional[Dict] = None) -> Optional[BaseSSOProvider]:
        """
        Get a singleton instance of the specified SSO provider.
        
        Args:
            provider_name: Name of the provider (e.g., 'google')
            config: Optional configuration dict (if None, provider reads from env vars)
            
        Returns:
            Singleton instance of the provider, or None if provider not found
        """
        provider_name = provider_name.lower()
        
        # Double-checked locking pattern for thread safety
        if provider_name not in cls._instances:
            with cls._lock:
                # Check again inside lock (double-checked locking)
                if provider_name not in cls._instances:
                    provider_class = cls._providers.get(provider_name)
                    if not provider_class:
                        return None
                    
                    # Create singleton instance
                    cls._instances[provider_name] = provider_class(config)
        
        return cls._instances[provider_name]
    
    @classmethod
    def register_provider(cls, name: str, provider_class: Type[BaseSSOProvider]):
        """
        Register a new provider type.
        
        Args:
            name: Provider name (e.g., 'okta')
            provider_class: Provider class that extends BaseSSOProvider
        """
        with cls._lock:
            cls._providers[name.lower()] = provider_class
            # Remove existing instance if any (force recreation on next get)
            if name.lower() in cls._instances:
                del cls._instances[name.lower()]
    
    @classmethod
    def get_all_providers(cls) -> Dict[str, BaseSSOProvider]:
        """
        Get all registered provider instances.
        
        Returns:
            Dictionary mapping provider names to their singleton instances
        """
        # Initialize all registered providers
        for provider_name in cls._providers.keys():
            cls.get_provider(provider_name)
        
        return cls._instances.copy()
    
    @classmethod
    def reset(cls):
        """
        Reset the registry (useful for testing).
        """
        with cls._lock:
            cls._instances.clear()
