"""Redis cache utility for repository branches."""

import json
import redis
from typing import Optional, List, Dict, Any
from app.core.config_provider import config_provider
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


class BranchCache:
    """Redis cache manager for repository branches."""

    def __init__(self):
        """Initialize Redis client."""
        try:
            redis_url = config_provider.get_redis_url()
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            # Test connection
            self.redis_client.ping()
            self.available = True
            logger.info("BranchCache: ✅ Redis connection established and available")
        except Exception as e:
            logger.warning(f"BranchCache: Redis not available - {str(e)}")
            self.redis_client = None
            self.available = False

    def _get_cache_key(self, repo_name: str, search_query: Optional[str] = None) -> str:
        """
        Generate cache key for repository branches.
        
        Args:
            repo_name: Repository name (e.g., "owner/repo")
            search_query: Optional search query for filtered results
            
        Returns:
            Cache key string
        """
        if search_query:
            # Include search query in key for filtered results
            # Normalize search query (lowercase, strip whitespace)
            normalized_search = search_query.lower().strip()
            return f"repo_branches:{repo_name}:search:{normalized_search}"
        return f"repo_branches:{repo_name}"

    def get_branches(
        self, 
        repo_name: str, 
        search_query: Optional[str] = None
    ) -> Optional[List[str]]:
        """
        Get cached branches for a repository.
        
        Args:
            repo_name: Repository name (e.g., "owner/repo")
            search_query: Optional search query for filtered results
            
        Returns:
            List of branch names if cached, None otherwise
        """
        if not self.available:
            return None

        try:
            cache_key = self._get_cache_key(repo_name, search_query)
            cached_data = self.redis_client.get(cache_key)
            
            if cached_data:
                branches = json.loads(cached_data)
                logger.info(
                    f"BranchCache: ✅ CACHE HIT for {repo_name} "
                    f"(search={search_query or 'none'}, branches={len(branches)}, key={cache_key})"
                )
                return branches
            
            logger.info(f"BranchCache: ❌ CACHE MISS for {repo_name} (search={search_query or 'none'}, key={cache_key})")
            return None
            
        except Exception as e:
            logger.warning(f"BranchCache: Error reading from cache: {str(e)}")
            return None

    def set_branches(
        self,
        repo_name: str,
        branches: List[str],
        search_query: Optional[str] = None,
        ttl: int = 3600
    ) -> bool:
        """
        Cache branches for a repository.
        
        Args:
            repo_name: Repository name (e.g., "owner/repo")
            branches: List of branch names to cache
            search_query: Optional search query for filtered results
            ttl: Time to live in seconds (default: 1 hour)
            
        Returns:
            True if cached successfully, False otherwise
        """
        if not self.available:
            return False

        try:
            cache_key = self._get_cache_key(repo_name, search_query)
            branches_json = json.dumps(branches)
            
            self.redis_client.setex(cache_key, ttl, branches_json)
            logger.info(
                f"BranchCache: 💾 CACHED {len(branches)} branches for {repo_name} "
                f"(search={search_query or 'none'}, ttl={ttl}s, key={cache_key})"
            )
            return True
            
        except Exception as e:
            logger.warning(f"BranchCache: Error writing to cache: {str(e)}")
            return False

    def cache_all_branches(
        self,
        repo_name: str,
        branches: List[str],
        ttl: int = 3600
    ) -> bool:
        """
        Cache all branches for a repository (without search filter).
        This is useful when fetching all branches - we cache them for future use.
        
        Args:
            repo_name: Repository name (e.g., "owner/repo")
            branches: List of all branch names
            ttl: Time to live in seconds (default: 1 hour)
            
        Returns:
            True if cached successfully, False otherwise
        """
        return self.set_branches(repo_name, branches, search_query=None, ttl=ttl)

    def invalidate_cache(self, repo_name: str) -> bool:
        """
        Invalidate all cached branches for a repository.
        
        Args:
            repo_name: Repository name (e.g., "owner/repo")
            
        Returns:
            True if invalidated successfully, False otherwise
        """
        if not self.available:
            return False

        try:
            # Delete both the main cache key and any search-filtered keys
            pattern = f"repo_branches:{repo_name}*"
            keys = self.redis_client.keys(pattern)
            
            if keys:
                self.redis_client.delete(*keys)
                logger.info(f"BranchCache: 🗑️ Invalidated cache for {repo_name} ({len(keys)} keys)")
            else:
                logger.info(f"BranchCache: No cache keys found for {repo_name}")
            
            return True
            
        except Exception as e:
            logger.warning(f"BranchCache: Error invalidating cache: {str(e)}")
            return False

    def get_cached_branches_with_pagination(
        self,
        repo_name: str,
        limit: Optional[int] = None,
        offset: int = 0,
        search_query: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached branches with pagination applied.
        
        Note: For search queries, we don't cache paginated results separately.
        Instead, we cache the full search result and apply pagination here.
        
        Args:
            repo_name: Repository name (e.g., "owner/repo")
            limit: Optional limit on number of branches to return
            offset: Number of branches to skip
            search_query: Optional search query
            
        Returns:
            Dictionary with branches, has_next_page, etc. if cached, None otherwise
        """
        cached_branches = self.get_branches(repo_name, search_query)
        
        if cached_branches is None:
            return None
        
        # Apply pagination to cached results
        paginated_branches = cached_branches[offset:]
        if limit is not None:
            paginated_branches = paginated_branches[:limit]
            has_next_page = (offset + limit) < len(cached_branches)
        else:
            has_next_page = False
        
        return {
            "branches": paginated_branches,
            "has_next_page": has_next_page,
            "total_count": len(cached_branches) if search_query else None
        }

    def get_cache_info(self, repo_name: str) -> Dict[str, Any]:
        """
        Get cache information for a repository (for debugging/verification).
        
        Args:
            repo_name: Repository name (e.g., "owner/repo")
            
        Returns:
            Dictionary with cache status, keys, and TTL information
        """
        if not self.available:
            return {
                "available": False,
                "message": "Redis not available"
            }
        
        try:
            pattern = f"repo_branches:{repo_name}*"
            keys = self.redis_client.keys(pattern)
            
            cache_info = {
                "available": True,
                "repo_name": repo_name,
                "keys_found": len(keys),
                "keys": []
            }
            
            for key in keys:
                ttl = self.redis_client.ttl(key)
                cached_data = self.redis_client.get(key)
                branch_count = 0
                if cached_data:
                    try:
                        branches = json.loads(cached_data)
                        branch_count = len(branches)
                    except:
                        pass
                
                cache_info["keys"].append({
                    "key": key,
                    "ttl_seconds": ttl,
                    "branch_count": branch_count
                })
            
            return cache_info
            
        except Exception as e:
            return {
                "available": True,
                "error": str(e)
            }
