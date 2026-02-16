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

# Cloudflare API base (constant)
CLOUDFLARE_API_BASE = "https://api.cloudflare.com/client/v4"

# Tunnel naming convention
TUNNEL_NAME_PREFIX = "potpie-user"

# Default local port for ingress (must match the port extension runs cloudflared with: --url http://localhost:PORT)
# Overridable via CLOUDFLARE_TUNNEL_INGRESS_PORT or by passing local_port in provision request.
DEFAULT_INGRESS_PORT = 3001


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
        # Read env at instantiation so we always see current values (e.g. after load_dotenv)
        self.account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID") or None
        self.api_token = os.getenv("CLOUDFLARE_API_TOKEN") or None
        self.tunnel_domain = (os.getenv("CLOUDFLARE_TUNNEL_DOMAIN") or "").strip() or None
        self.headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }

        if not self.account_id or not self.api_token:
            logger.warning(
                "[CloudflareTunnel] Missing CLOUDFLARE_ACCOUNT_ID or CLOUDFLARE_API_TOKEN. "
                "Named tunnels will not be available. Extension will use quick tunnels."
            )

    def is_configured(self) -> bool:
        """Check if Cloudflare API is configured."""
        return bool(self.account_id and self.api_token)

    def has_domain(self) -> bool:
        """Check if a domain is configured for ingress."""
        return bool(self.tunnel_domain)

    def _ingress_port(self, local_port: Optional[int] = None) -> int:
        """Resolve local port for ingress: request > env > default."""
        if local_port is not None and 1 <= local_port <= 65535:
            return local_port
        env_port = os.getenv("CLOUDFLARE_TUNNEL_INGRESS_PORT", "").strip()
        if env_port and env_port.isdigit():
            return int(env_port)
        return DEFAULT_INGRESS_PORT

    async def provision_tunnel_for_user(
        self, user_id: str, local_port: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Create or get existing named tunnel for a user.

        Args:
            user_id: The user's unique identifier
            local_port: Optional port the extension will use (--url http://localhost:PORT). Ingress will use this so cloudflared forwards to the correct port.

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
                    logger.info(
                        f"[CloudflareTunnel] Found existing tunnel: {tunnel_name}"
                    )
                    tunnel_id = existing["id"]

                    # Ensure ingress is configured if domain is available; use our result
                    # so we don't rely on a possibly stale config GET in _get_tunnel_credentials
                    known_ingress = False
                    known_tunnel_url = None
                    if self.has_domain():
                        subdomain = f"tunnel-{user_id[:8]}"
                        port = self._ingress_port(local_port)
                        known_ingress = await self._configure_tunnel_ingress(
                            client, tunnel_id, subdomain, local_port=port
                        )
                        if not known_ingress:
                            logger.warning(
                                f"[CloudflareTunnel] Failed to configure ingress for existing tunnel {tunnel_id}"
                            )
                        else:
                            known_tunnel_url = f"https://{subdomain}.{self.tunnel_domain}"

                    credentials = await self._get_tunnel_credentials(
                        client, tunnel_id, tunnel_name
                    )
                    if credentials and known_ingress and known_tunnel_url:
                        credentials["ingress_configured"] = True
                        credentials["tunnel_url"] = known_tunnel_url
                    return credentials

                # Create new tunnel
                # For remotely-managed tunnels, we use config_src: "cloudflare" and configure ingress via API
                # Reference: https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/get-started/create-remote-tunnel/
                response = await client.post(
                    f"{CLOUDFLARE_API_BASE}/accounts/{self.account_id}/cfd_tunnel",
                    headers=self.headers,
                    json={
                        "name": tunnel_name,
                        "config_src": "cloudflare",  # Remotely-managed, requires ingress config
                    },
                    timeout=30.0,
                )

                if response.status_code != 200:
                    logger.error(
                        f"[CloudflareTunnel] Failed to create tunnel: {response.status_code} - {response.text}"
                    )
                    return None

                data = response.json()
                if not data.get("success"):
                    logger.error(f"[CloudflareTunnel] API error: {data.get('errors')}")
                    return None

                tunnel = data["result"]
                tunnel_id = tunnel["id"]

                logger.info(
                    f"[CloudflareTunnel] Created new tunnel: {tunnel_name} ({tunnel_id})"
                )

                # Configure ingress if domain is available
                ingress_configured = False
                if self.has_domain():
                    subdomain = f"tunnel-{user_id[:8]}"
                    port = self._ingress_port(local_port)
                    ingress_configured = await self._configure_tunnel_ingress(
                        client, tunnel_id, subdomain, local_port=port
                    )
                    if not ingress_configured:
                        logger.warning(
                            f"[CloudflareTunnel] Failed to configure ingress for tunnel {tunnel_id}"
                        )

                # Get token
                credentials = await self._get_tunnel_credentials(
                    client, tunnel_id, tunnel_name
                )
                if credentials:
                    credentials["ingress_configured"] = ingress_configured
                    if ingress_configured:
                        # Use domain-based URL if ingress is configured
                        subdomain = f"tunnel-{user_id[:8]}"
                        credentials["tunnel_url"] = (
                            f"https://{subdomain}.{self.tunnel_domain}"
                        )

                return credentials

        except httpx.TimeoutException:
            logger.error("[CloudflareTunnel] Timeout while provisioning tunnel")
            return None
        except Exception as e:
            logger.error(f"[CloudflareTunnel] Error provisioning tunnel: {e}")
            return None

    async def _find_tunnel_by_name(
        self, client: httpx.AsyncClient, name: str
    ) -> Optional[Dict]:
        """Find existing tunnel by name (case-insensitive, partial match)."""
        try:
            # First try exact match
            response = await client.get(
                f"{CLOUDFLARE_API_BASE}/accounts/{self.account_id}/cfd_tunnel",
                headers=self.headers,
                params={"name": name, "is_deleted": False},
                timeout=30.0,
            )

            if response.status_code == 200:
                data = response.json()
                tunnels = data.get("result", [])
                if tunnels:
                    logger.info(
                        f"[CloudflareTunnel] Found tunnel by exact name match: {name}"
                    )
                    return tunnels[0]

            # If exact match fails, list all tunnels and search (case-insensitive)
            logger.info(
                f"[CloudflareTunnel] Exact match failed, searching all tunnels for: {name}"
            )
            response = await client.get(
                f"{CLOUDFLARE_API_BASE}/accounts/{self.account_id}/cfd_tunnel",
                headers=self.headers,
                params={"is_deleted": False},
                timeout=30.0,
            )

            if response.status_code == 200:
                data = response.json()
                all_tunnels = data.get("result", [])
                logger.info(
                    f"[CloudflareTunnel] Found {len(all_tunnels)} total tunnel(s)"
                )

                # Try case-insensitive partial match
                name_lower = name.lower()
                for tunnel in all_tunnels:
                    tunnel_name = tunnel.get("name", "")
                    if (
                        name_lower in tunnel_name.lower()
                        or tunnel_name.lower() in name_lower
                    ):
                        logger.info(
                            f"[CloudflareTunnel] Found tunnel by partial match: {tunnel_name} (searching for: {name})"
                        )
                        return tunnel

                # If still no match, return first tunnel with ingress configured (fallback)
                logger.warning(
                    f"[CloudflareTunnel] No name match found, checking for tunnels with ingress..."
                )
                for tunnel in all_tunnels:
                    tunnel_id = tunnel.get("id")
                    if tunnel_id:
                        # Quick check if this tunnel has ingress
                        config_resp = await client.get(
                            f"{CLOUDFLARE_API_BASE}/accounts/{self.account_id}/cfd_tunnel/{tunnel_id}/configurations",
                            headers=self.headers,
                            timeout=10.0,
                        )
                        if config_resp.status_code == 200:
                            config_data = config_resp.json()
                            if config_data.get("success"):
                                result = config_data.get("result")
                                if result is None:
                                    continue
                                config = (
                                    result.get("config")
                                    if isinstance(result, dict)
                                    else {}
                                )
                                if config is None:
                                    config = {}
                                ingress = config.get("ingress", [])
                                # Check if any ingress rule has a hostname
                                for rule in ingress:
                                    if rule.get("hostname"):
                                        logger.info(
                                            f"[CloudflareTunnel] Found tunnel with configured hostname: {tunnel.get('name')} ({tunnel_id})"
                                        )
                                        return tunnel

            return None
        except Exception as e:
            logger.error(f"[CloudflareTunnel] Error finding tunnel: {e}")
            return None

    async def _configure_tunnel_ingress(
        self,
        client: httpx.AsyncClient,
        tunnel_id: str,
        subdomain: Optional[str] = None,
        local_port: Optional[int] = None,
    ) -> bool:
        """
        Configure ingress rules for the tunnel.

        This sets up routing so traffic to the tunnel hostname is forwarded to localhost:PORT.
        The port must match what the extension uses with cloudflared: --url http://localhost:PORT.
        Requires CLOUDFLARE_TUNNEL_DOMAIN to be set.
        """
        if not self.has_domain():
            return False

        port = local_port if local_port is not None else self._ingress_port(None)
        service_url = f"http://localhost:{port}"

        try:
            # Build ingress: hostname rule so named tunnel has a public URL
            hostname = (
                f"{subdomain}.{self.tunnel_domain}"
                if subdomain
                else None
            )
            if hostname:
                ingress = [
                    {
                        "hostname": hostname,
                        "service": service_url,
                    },
                    {"service": "http_status:404"},  # Catch-all for unmatched routes
                ]
                logger.info(
                    f"[CloudflareTunnel] Configuring ingress with hostname: {hostname}, service: {service_url}"
                )
            else:
                ingress = [
                    {"service": service_url},
                    {"service": "http_status:404"},
                ]
            config = {"ingress": ingress}

            response = await client.put(
                f"{CLOUDFLARE_API_BASE}/accounts/{self.account_id}/cfd_tunnel/{tunnel_id}/configurations",
                headers=self.headers,
                json={"config": config},
                timeout=30.0,
            )

            if response.status_code != 200:
                logger.error(
                    f"[CloudflareTunnel] Failed to configure ingress: {response.status_code} - {response.text}"
                )
                return False

            data = response.json()
            if not data.get("success"):
                logger.error(
                    f"[CloudflareTunnel] Ingress config error: {data.get('errors')}"
                )
                return False

            logger.info(
                f"[CloudflareTunnel] ✅ Configured ingress for tunnel {tunnel_id}"
            )
            return True

        except Exception as e:
            logger.error(f"[CloudflareTunnel] Error configuring ingress: {e}")
            return False

    async def _get_tunnel_credentials(
        self, client: httpx.AsyncClient, tunnel_id: str, tunnel_name: str
    ) -> Optional[Dict[str, Any]]:
        """Get the tunnel token for cloudflared to use."""
        try:
            response = await client.get(
                f"{CLOUDFLARE_API_BASE}/accounts/{self.account_id}/cfd_tunnel/{tunnel_id}/token",
                headers=self.headers,
                timeout=30.0,
            )

            if response.status_code != 200:
                logger.error(
                    f"[CloudflareTunnel] Failed to get token: {response.status_code}"
                )
                return None

            data = response.json()
            if not data.get("success"):
                logger.error(
                    f"[CloudflareTunnel] Token API error: {data.get('errors')}"
                )
                return None

            token = data["result"]

            # Check if ingress is configured by checking tunnel config
            ingress_configured = False
            tunnel_url = None

            config_response = await client.get(
                f"{CLOUDFLARE_API_BASE}/accounts/{self.account_id}/cfd_tunnel/{tunnel_id}/configurations",
                headers=self.headers,
                timeout=30.0,
            )

            if config_response.status_code == 200:
                config_data = config_response.json()
                logger.debug(f"[CloudflareTunnel] Config response: {config_data}")

                if config_data.get("success"):
                    result = config_data.get("result")
                    if result is None:
                        logger.warning(
                            f"[CloudflareTunnel] Config API returned success=True but result is None"
                        )
                        config = {}
                    else:
                        config = (
                            result.get("config") if isinstance(result, dict) else {}
                        )
                        if config is None:
                            config = {}
                    ingress_rules = config.get("ingress", [])

                    logger.info(
                        f"[CloudflareTunnel] Config structure: {list(config.keys())}"
                    )
                    logger.info(
                        f"[CloudflareTunnel] Ingress rules found: {len(ingress_rules) if ingress_rules else 0}"
                    )

                    if ingress_rules:
                        ingress_configured = True
                        logger.info(
                            f"[CloudflareTunnel] Found {len(ingress_rules)} ingress rule(s) for tunnel {tunnel_id}"
                        )
                        logger.debug(
                            f"[CloudflareTunnel] Ingress rules: {ingress_rules}"
                        )

                        # Try to find a public hostname from ingress rules
                        # Look for routes with hostname (published applications)
                        for rule in ingress_rules:
                            hostname = rule.get("hostname")
                            logger.debug(
                                f"[CloudflareTunnel] Checking rule: {rule}, hostname: {hostname}"
                            )
                            if hostname:
                                tunnel_url = f"https://{hostname}"
                                logger.info(
                                    f"[CloudflareTunnel] ✅ Using configured hostname: {hostname}"
                                )
                                break

                        # If no hostname found, check if domain is set and construct URL
                        if not tunnel_url and self.has_domain():
                            # Extract user_id from tunnel_name (potpie-user-{user_id[:8]})
                            user_part = tunnel_name.replace(
                                f"{TUNNEL_NAME_PREFIX}-", ""
                            )
                            subdomain = f"tunnel-{user_part}"
                            tunnel_url = (
                                f"https://{subdomain}.{self.tunnel_domain}"
                            )
                            logger.info(
                                f"[CloudflareTunnel] Constructed URL from domain: {tunnel_url}"
                            )
                    else:
                        logger.warning(
                            f"[CloudflareTunnel] ⚠️ No ingress rules found in config"
                        )
                else:
                    logger.warning(
                        f"[CloudflareTunnel] ⚠️ Config API returned success=False: {config_data.get('errors')}"
                    )
            else:
                logger.error(
                    f"[CloudflareTunnel] Failed to get config: {config_response.status_code} - {config_response.text}"
                )

            result = {
                "tunnel_id": tunnel_id,
                "tunnel_name": tunnel_name,
                "tunnel_token": token,
                "ingress_configured": ingress_configured,
            }

            # Set URL
            if tunnel_url:
                result["tunnel_url"] = tunnel_url
            elif ingress_configured:
                # Ingress configured but no hostname found - use cfargotunnel.com as fallback
                result["tunnel_url"] = f"https://{tunnel_id}.cfargotunnel.com"
                logger.warning(
                    f"[CloudflareTunnel] Ingress configured but no hostname found, using fallback URL"
                )
            else:
                # No ingress - won't work, but return URL for fallback detection
                result["tunnel_url"] = f"https://{tunnel_id}.cfargotunnel.com"

            return result

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
                    timeout=30.0,
                )

                if response.status_code == 200:
                    data = response.json()
                    tunnel = data.get("result", {})
                    return {
                        "id": tunnel.get("id"),
                        "name": tunnel.get("name"),
                        "status": tunnel.get("status"),
                        "connections": len(tunnel.get("connections", [])),
                        "created_at": tunnel.get("created_at"),
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
                    timeout=30.0,
                )

                # Then delete the tunnel
                response = await client.delete(
                    f"{CLOUDFLARE_API_BASE}/accounts/{self.account_id}/cfd_tunnel/{tunnel_id}",
                    headers=self.headers,
                    timeout=30.0,
                )

                if response.status_code == 200:
                    logger.info(f"[CloudflareTunnel] Deleted tunnel: {tunnel_id}")
                    return True
                else:
                    logger.error(
                        f"[CloudflareTunnel] Failed to delete: {response.text}"
                    )
                    return False

        except Exception as e:
            logger.error(f"[CloudflareTunnel] Error deleting tunnel: {e}")
            return False


def get_cloudflare_tunnel_service() -> CloudflareTunnelService:
    """Return a CloudflareTunnelService that reads env at call time (no singleton)."""
    return CloudflareTunnelService()
