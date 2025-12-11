# Integrations module for external service integrations
import hashlib


def hash_user_id(user_id: str) -> str:
    """
    Hash user ID for safe logging (first 8 chars of SHA256).
    
    This function provides consistent user ID hashing across all integration modules
    to prevent exposing user IDs in logs while still allowing correlation.
    
    Args:
        user_id: The user ID to hash
        
    Returns:
        First 8 characters of SHA256 hash of the user ID
    """
    if not user_id:
        return "unknown"
    return hashlib.sha256(user_id.encode()).hexdigest()[:8]
