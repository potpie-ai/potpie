"""
Cloudflare Named Tunnel Service

Manages persistent named tunnels for users via Cloudflare API.
More reliable than quick tunnels - URLs don't change on restart.
"""

import os
import secrets
from typing import Optional, Dict, Any

import httpx
from loguru import logger

# Cloudflare API credentials (set in environment)
CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID")
CLOUDFLARE_API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN")
CLOUDFLARE_API_BASE = "https://api.cloudflare.com/client/v4"

# Tunnel naming convention
TUNNEL_NAME_PREFIX = "potpie-user"


class CloudflareTunnelService:
    """
    Service for managing Cloudflare named tunnels.
    
    Named tunnels provide:
    - Persistent URLs (don't change on restart)
    - Better reliability (99%+ uptime)
    - Cloudflare dashboard visibility
    - Auto-reconnection support
    """
    
    def __init__(self):
        self.account_id = CLOUDFLARE_ACCOUNT_ID
        self.api_token = CLOUDFLARE_API_TOKEN
        self.headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        
        if not self.account_id or not self.api_token:
            logger.warning(
                "[CloudflareTunnel] Missing CLOUDFLARE_ACCOUNT_ID or CLOUDFLARE_API_TOKEN. "
                "Named tunnels will not be available. Extension will use quick tunnels."
            )
    
    def is_configured(self) -> bool:
        """Check if Cloudflare API is configured."""
        return bool(self.account_id and self.api_token)
    
    async def provision_tunnel_for_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Create or get existing named tunnel for a user.
        
        Args:
            user_id: The user's unique identifier
            
        Returns:
            {
                "tunnel_id": "uuid",
                "tunnel_name": "potpie-user-abc123",
                "tunnel_token": "eyJ...",
                "tunnel_url": "https://{tunnel_id}.cfargotunnel.com"
            }
            or None on failure
        """
        if not self.is_configured():
            logger.error("[CloudflareTunnel] Not configured, cannot provision tunnel")
            return None
        
        # Create a unique but readable tunnel name
        tunnel_name = f"{TUNNEL_NAME_PREFIX}-{user_id[:8]}"
        
        try:
            async with httpx.AsyncClient() as client:
                # Check if tunnel already exists for this user
                existing = await self._find_tunnel_by_name(client, tunnel_name)
                if existing:
                    logger.info(f"[CloudflareTunnel] Found existing tunnel: {tunnel_name}")
                    return await self._get_tunnel_credentials(client, existing["id"], tunnel_name)
                
                # Create new tunnel with a random secret
                tunnel_secret = secrets.token_bytes(32).hex()
                
                response = await client.post(
                    f"{CLOUDFLARE_API_BASE}/accounts/{self.account_id}/cfd_tunnel",
                    headers=self.headers,
                    json={
                        "name": tunnel_name,
                        "tunnel_secret": tunnel_secret,
                        "config_src": "local"  # Config managed by cloudflared
                    },
                    timeout=30.0
                )
                
                if response.status_code != 200:
                    logger.error(f"[CloudflareTunnel] Failed to create tunnel: {response.status_code} - {response.text}")
                    return None
                
                data = response.json()
                if not data.get("success"):
                    logger.error(f"[CloudflareTunnel] API error: {data.get('errors')}")
                    return None
                
                tunnel = data["result"]
                tunnel_id = tunnel["id"]
                
                logger.info(f"[CloudflareTunnel] Created new tunnel: {tunnel_name} ({tunnel_id})")
                
                return await self._get_tunnel_credentials(client, tunnel_id, tunnel_name)
                
        except httpx.TimeoutException:
            logger.error("[CloudflareTunnel] Timeout while provisioning tunnel")
            return None
        except Exception as e:
            logger.error(f"[CloudflareTunnel] Error provisioning tunnel: {e}")
            return None
    
    async def _find_tunnel_by_name(self, client: httpx.AsyncClient, name: str) -> Optional[Dict]:
        """Find existing tunnel by name."""
        try:
            response = await client.get(
                f"{CLOUDFLARE_API_BASE}/accounts/{self.account_id}/cfd_tunnel",
                headers=self.headers,
                params={"name": name, "is_deleted": False},
                timeout=30.0
            )
            
            if response.status_code == 200:
                data = response.json()
                tunnels = data.get("result", [])
                if tunnels:
                    return tunnels[0]
            return None
        except Exception as e:
            logger.error(f"[CloudflareTunnel] Error finding tunnel: {e}")
            return None
    
    async def _get_tunnel_credentials(
        self, 
        client: httpx.AsyncClient, 
        tunnel_id: str, 
        tunnel_name: str
    ) -> Optional[Dict[str, Any]]:
        """Get the tunnel token for cloudflared to use."""
        try:
            response = await client.get(
                f"{CLOUDFLARE_API_BASE}/accounts/{self.account_id}/cfd_tunnel/{tunnel_id}/token",
                headers=self.headers,
                timeout=30.0
            )
            
            if response.status_code != 200:
                logger.error(f"[CloudflareTunnel] Failed to get token: {response.status_code}")
                return None
            
            data = response.json()
            if not data.get("success"):
                logger.error(f"[CloudflareTunnel] Token API error: {data.get('errors')}")
                return None
            
            token = data["result"]
            
            return {
                "tunnel_id": tunnel_id,
                "tunnel_name": tunnel_name,
                "tunnel_token": token,
                "tunnel_url": f"https://{tunnel_id}.cfargotunnel.com"
            }
            
        except Exception as e:
            logger.error(f"[CloudflareTunnel] Error getting token: {e}")
            return None
    
    async def get_tunnel_status(self, tunnel_id: str) -> Optional[Dict[str, Any]]:
        """Get tunnel connection status from Cloudflare."""
        if not self.is_configured():
            return None
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{CLOUDFLARE_API_BASE}/accounts/{self.account_id}/cfd_tunnel/{tunnel_id}",
                    headers=self.headers,
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    tunnel = data.get("result", {})
                    return {
                        "id": tunnel.get("id"),
                        "name": tunnel.get("name"),
                        "status": tunnel.get("status"),
                        "connections": len(tunnel.get("connections", [])),
                        "created_at": tunnel.get("created_at")
                    }
                return None
                
        except Exception as e:
            logger.error(f"[CloudflareTunnel] Error getting status: {e}")
            return None
    
    async def delete_tunnel(self, tunnel_id: str) -> bool:
        """
        Delete a tunnel (cleanup).
        
        Note: Must disconnect all connections first.
        """
        if not self.is_configured():
            return False
        
        try:
            async with httpx.AsyncClient() as client:
                # First, clean up any active connections
                await client.delete(
                    f"{CLOUDFLARE_API_BASE}/accounts/{self.account_id}/cfd_tunnel/{tunnel_id}/connections",
                    headers=self.headers,
                    timeout=30.0
                )
                
                # Then delete the tunnel
                response = await client.delete(
                    f"{CLOUDFLARE_API_BASE}/accounts/{self.account_id}/cfd_tunnel/{tunnel_id}",
                    headers=self.headers,
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    logger.info(f"[CloudflareTunnel] Deleted tunnel: {tunnel_id}")
                    return True
                else:
                    logger.error(f"[CloudflareTunnel] Failed to delete: {response.text}")
                    return False
                    
        except Exception as e:
            logger.error(f"[CloudflareTunnel] Error deleting tunnel: {e}")
            return False


# Singleton instance
_cloudflare_service: Optional[CloudflareTunnelService] = None


def get_cloudflare_tunnel_service() -> CloudflareTunnelService:
    """Get the singleton CloudflareTunnelService instance."""
    global _cloudflare_service
    if _cloudflare_service is None:
        _cloudflare_service = CloudflareTunnelService()
    return _cloudflare_service
