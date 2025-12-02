"""
Base SSO Provider Interface

All SSO providers must implement this interface.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class SSOUserInfo:
    """Standardized user information from SSO provider"""
    
    email: str
    email_verified: bool
    provider_uid: str  # Unique ID from provider
    display_name: Optional[str] = None
    given_name: Optional[str] = None
    family_name: Optional[str] = None
    picture: Optional[str] = None
    raw_data: Optional[Dict[str, Any]] = None  # Full provider response


class BaseSSOProvider(ABC):
    """
    Base class for all SSO providers.
    
    Each provider must implement:
    - Token verification
    - User info extraction
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize provider with configuration.
        
        Args:
            config: Provider-specific configuration (client_id, client_secret, etc.)
        """
        self.config = config or {}
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name (e.g., 'google', 'azure')"""
        pass
    
    @abstractmethod
    async def verify_token(self, id_token: str) -> SSOUserInfo:
        """
        Verify an ID token and extract user information.
        
        Args:
            id_token: The ID token from the provider
            
        Returns:
            SSOUserInfo with verified user data
            
        Raises:
            ValueError: If token is invalid or expired
        """
        pass
    
    @abstractmethod
    def get_authorization_url(
        self,
        redirect_uri: str,
        state: Optional[str] = None,
        scopes: Optional[list] = None,
    ) -> str:
        """
        Generate authorization URL for OAuth flow.
        
        Args:
            redirect_uri: Where to redirect after authorization
            state: CSRF protection state
            scopes: Requested scopes
            
        Returns:
            Authorization URL to redirect user to
        """
        pass
    
    def validate_config(self) -> bool:
        """
        Validate that provider configuration is correct.
        
        Returns:
            True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        required_keys = self.get_required_config_keys()
        
        for key in required_keys:
            if key not in self.config:
                raise ValueError(
                    f"Missing required configuration key '{key}' for {self.provider_name}"
                )
        
        return True
    
    @abstractmethod
    def get_required_config_keys(self) -> list:
        """Return list of required configuration keys"""
        pass

