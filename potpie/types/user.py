"""User-related type definitions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class UserInfo:
    """Information about a user."""

    uid: str
    email: str
    display_name: Optional[str] = None
    email_verified: bool = False
    created_at: Optional[datetime] = None
    last_login_at: Optional[datetime] = None
    provider_username: Optional[str] = None

    @classmethod
    def from_model(cls, user) -> UserInfo:
        """Create UserInfo from User model instance."""
        return cls(
            uid=user.uid,
            email=user.email,
            display_name=user.display_name,
            email_verified=user.email_verified,
            created_at=user.created_at,
            last_login_at=user.last_login_at,
            provider_username=user.provider_username,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "uid": self.uid,
            "email": self.email,
            "display_name": self.display_name,
            "email_verified": self.email_verified,
            "created_at": self.created_at,
            "last_login_at": self.last_login_at,
            "provider_username": self.provider_username,
        }
