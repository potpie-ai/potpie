"""
Repo Manager Module

Service for managing user repositories, tracking availability, and handling eviction.
"""

from .repo_manager_interface import IRepoManager
from .repo_manager import RepoManager
from .sync_helper import ensure_repo_registered

__all__ = ["IRepoManager", "RepoManager", "ensure_repo_registered"]
