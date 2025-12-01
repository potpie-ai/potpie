"""
Repo Manager Module

Service for managing user repositories, tracking availability, and handling eviction.
"""

from .repo_manager_interface import IRepoManager
from .repo_manager import RepoManager

__all__ = ["IRepoManager", "RepoManager"]
