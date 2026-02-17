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

# Tunnel config: only .../configurations exists in the Cloudflare API (PUT updates tunnel ingress).
# There is no .../config endpoint in client v4; the official SDK uses "configurations".
CFD_TUNNEL_CONFIG_PATH = "configurations"

# Zero Trust Networks: create hostname route so it appears in dashboard "Hostname routes".
# POST accounts/{account_id}/networks/hostname_routes (optional; tunnel ingress already routes traffic).
NETWORKS_HOSTNAME_ROUTES_PATH = "networks/hostname_routes"

# Tunnel naming convention
TUNNEL_NAME_PREFIX = "potpie-user"
WORKSPACE_TUNNEL_NAME_PREFIX = "potpie-ws"

# Default local port for ingress (must match the port extension runs cloudflared with: --url http://localhost:PORT)
# Overridable via CLOUDFLARE_TUNNEL_INGRESS_PORT or by passing local_port in provision request.
DEFAULT_INGRESS_PORT = 3001

# When set, workspace tunnel URL is https://{workspace_id}.{TUNNEL_WILDCARD_DOMAIN}; must match router so ingress hostname is registered.
TUNNEL_WILDCARD_DOMAIN_ENV = "TUNNEL_WILDCARD_DOMAIN"


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

    def _workspace_tunnel_public_hostname(
        self, workspace_id: str, tunnel_id: str
    ) -> str:
        """Hostname we return in tunnel_url; must be registered in tunnel config so Cloudflare routes to this tunnel."""
        wildcard = (os.getenv(TUNNEL_WILDCARD_DOMAIN_ENV) or "").strip() or None
        if wildcard:
            return f"{workspace_id}.{wildcard}"
        return f"{tunnel_id}.cfargotunnel.com"

    async def provision_workspace_tunnel(
        self, workspace_id: str, local_port: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Create a workspace-scoped tunnel: potpie-ws-{workspace_id}.
        Router Service proxies to https://{tunnel_id}.cfargotunnel.com. The tunnel's
        ingress is set to origin http://127.0.0.1:<local_port> via Cloudflare API so
        token-based (remotely-managed) runs work; without this, tunnel URL would 404.

        Returns:
            {"tunnel_id": "uuid", "tunnel_name": "potpie-ws-...", "tunnel_token": "eyJ..."}
            or None on failure
        """
        if not self.is_configured():
            logger.error("[CloudflareTunnel] Not configured, cannot provision workspace tunnel")
            return None
        tunnel_name = f"{WORKSPACE_TUNNEL_NAME_PREFIX}-{workspace_id}"
        try:
            async with httpx.AsyncClient() as client:
                existing = await self._find_tunnel_by_name(
                    client, tunnel_name, allow_ingress_fallback=False
                )
                if existing:
                    tunnel_id = existing["id"]
                    hostname = self._workspace_tunnel_public_hostname(
                        workspace_id, tunnel_id
                    )
                    await self._configure_tunnel_ingress_public_hostname(
                        client, tunnel_id, hostname, local_port
                    )
                    credentials = await self._get_tunnel_credentials(
                        client, tunnel_id, tunnel_name
                    )
                    if credentials:
                        return {
                            "tunnel_id": tunnel_id,
                            "tunnel_name": tunnel_name,
                            "tunnel_token": credentials["tunnel_token"],
                        }
                response = await client.post(
                    f"{CLOUDFLARE_API_BASE}/accounts/{self.account_id}/cfd_tunnel",
                    headers=self.headers,
                    json={
                        "name": tunnel_name,
                        "config_src": "cloudflare",
                    },
                    timeout=30.0,
                )
                if response.status_code == 409:
                    # Another request created it; re-fetch by name and return credentials (idempotent)
                    logger.info(
                        f"[CloudflareTunnel] Tunnel already exists (409), re-fetching: {tunnel_name}"
                    )
                    existing = await self._find_tunnel_by_name(
                        client, tunnel_name, allow_ingress_fallback=False
                    )
                    if existing:
                        tunnel_id = existing["id"]
                        hostname = self._workspace_tunnel_public_hostname(
                            workspace_id, tunnel_id
                        )
                        await self._configure_tunnel_ingress_public_hostname(
                            client, tunnel_id, hostname, local_port
                        )
                        credentials = await self._get_tunnel_credentials(
                            client, tunnel_id, tunnel_name
                        )
                        if credentials:
                            return {
                                "tunnel_id": tunnel_id,
                                "tunnel_name": tunnel_name,
                                "tunnel_token": credentials["tunnel_token"],
                            }
                    logger.error(
                        f"[CloudflareTunnel] 409 but could not re-fetch tunnel: {tunnel_name}"
                    )
                    return None
                if response.status_code != 200:
                    logger.error(
                        f"[CloudflareTunnel] Failed to create workspace tunnel: {response.status_code} - {response.text}"
                    )
                    return None
                data = response.json()
                if not data.get("success"):
                    logger.error(f"[CloudflareTunnel] API error: {data.get('errors')}")
                    return None
                tunnel = data["result"]
                tunnel_id = tunnel["id"]
                logger.info(
                    f"[CloudflareTunnel] Created workspace tunnel: {tunnel_name} ({tunnel_id})"
                )
                hostname = self._workspace_tunnel_public_hostname(
                    workspace_id, tunnel_id
                )
                await self._configure_tunnel_ingress_public_hostname(
                    client, tunnel_id, hostname, local_port
                )
                credentials = await self._get_tunnel_credentials(
                    client, tunnel_id, tunnel_name
                )
                if not credentials:
                    return None
                return {
                    "tunnel_id": tunnel_id,
                    "tunnel_name": tunnel_name,
                    "tunnel_token": credentials["tunnel_token"],
                }
        except httpx.TimeoutException:
            logger.error("[CloudflareTunnel] Timeout while provisioning workspace tunnel")
            return None
        except Exception as e:
            logger.error(f"[CloudflareTunnel] Error provisioning workspace tunnel: {e}")
            return None

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
        self,
        client: httpx.AsyncClient,
        name: str,
        *,
        allow_ingress_fallback: bool = True,
    ) -> Optional[Dict]:
        """
        Find existing tunnel by name (exact, then case-insensitive partial match).
        If allow_ingress_fallback is True and no name match, returns first tunnel with
        ingress (for legacy user-level flow). Set to False for workspace tunnels so we
        only reuse potpie-ws-{id} or create new, never return a different tunnel.
        """
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

                # If still no match and fallback allowed, return first tunnel with ingress (legacy user-level only)
                if not allow_ingress_fallback:
                    logger.debug(
                        f"[CloudflareTunnel] No name match for {name}; not using ingress fallback (workspace tunnel)"
                    )
                    return None
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

    async def _put_tunnel_config(
        self,
        client: httpx.AsyncClient,
        tunnel_id: str,
        config: Dict[str, Any],
    ) -> tuple[bool, Optional[str]]:
        """
        PUT tunnel config (ingress) via the only Cloudflare API endpoint:
        PUT .../cfd_tunnel/{tunnel_id}/configurations. This sets the tunnel's
        ingress (public hostnames + origin). Returns (success, error_message).
        """
        body = {"config": config}
        url = f"{CLOUDFLARE_API_BASE}/accounts/{self.account_id}/cfd_tunnel/{tunnel_id}/{CFD_TUNNEL_CONFIG_PATH}"
        try:
            response = await client.put(
                url,
                headers=self.headers,
                json=body,
                timeout=30.0,
            )
            if response.status_code != 200:
                return False, f"{response.status_code} {response.text}"
            data = response.json()
            if not data.get("success"):
                return False, str(data.get("errors", "unknown"))
            logger.info(
                f"[CloudflareTunnel] PUT {CFD_TUNNEL_CONFIG_PATH} succeeded for tunnel {tunnel_id}"
            )
            return True, None
        except Exception as e:
            return False, str(e)

    async def _configure_tunnel_ingress_origin_only(
        self,
        client: httpx.AsyncClient,
        tunnel_id: str,
        local_port: Optional[int] = None,
    ) -> bool:
        """
        Configure tunnel ingress to origin only (no custom hostname).
        Used for workspace tunnels: traffic to https://{tunnel_id}.cfargotunnel.com
        is forwarded to http://127.0.0.1:<local_port>. Required for token-based
        (remotely-managed) tunnels so the edge has a route; without this, tunnel URL returns 404.
        """
        port = self._ingress_port(local_port)
        service_url = f"http://127.0.0.1:{port}"
        try:
            ingress = [
                {"service": service_url},
                {"service": "http_status:404"},
            ]
            config = {"ingress": ingress}
            ok, err = await self._put_tunnel_config(client, tunnel_id, config)
            if not ok:
                logger.error(
                    f"[CloudflareTunnel] Failed to configure origin-only ingress: {err}"
                )
                return False
            logger.info(
                f"[CloudflareTunnel] ✅ Configured origin-only ingress for tunnel {tunnel_id} -> {service_url}"
            )
            return True
        except Exception as e:
            logger.error(f"[CloudflareTunnel] Error configuring origin-only ingress: {e}")
            return False

    async def _configure_tunnel_ingress_public_hostname(
        self,
        client: httpx.AsyncClient,
        tunnel_id: str,
        hostname: str,
        local_port: Optional[int] = None,
        *,
        also_register_cfargotunnel: bool = True,
    ) -> bool:
        """
        Configure tunnel ingress with a public hostname (Put Tunnel Configuration).
        This populates "Hostname routes" / "Public Hostname" in the Cloudflare dashboard
        so the tunnel_url we return actually routes traffic. Required for token-based
        (remotely-managed) tunnels: without this, the connector can register but the
        hostname has no route and requests 404.

        When using the wildcard router, the backend proxies to https://{tunnel_id}.cfargotunnel.com
        (Host is that), so we must also register that hostname in ingress; otherwise the request
        hits the catch-all and returns 404.
        Ingress: rule(s) for hostname and optionally tunnel_id.cfargotunnel.com, then catch-all http_status:404.
        """
        port = self._ingress_port(local_port)
        service_url = f"http://127.0.0.1:{port}"
        cfargotunnel_host = f"{tunnel_id}.cfargotunnel.com"
        try:
            ingress_rules = []
            if hostname and hostname != cfargotunnel_host:
                ingress_rules.append({"hostname": hostname, "service": service_url})
            if also_register_cfargotunnel:
                ingress_rules.append({"hostname": cfargotunnel_host, "service": service_url})
            ingress = ingress_rules + [{"service": "http_status:404"}]
            config = {"ingress": ingress}
            ok, err = await self._put_tunnel_config(client, tunnel_id, config)
            if not ok:
                logger.error(
                    f"[CloudflareTunnel] Failed to configure ingress with hostname: {err}"
                )
                return False
            # Optionally create Zero Trust Networks hostname route so it shows in dashboard "Hostname routes".
            if hostname and hostname != cfargotunnel_host:
                await self._create_network_hostname_route(
                    client, hostname, tunnel_id, service_url
                )
            logger.info(
                f"[CloudflareTunnel] ✅ Configured ingress for tunnel {tunnel_id} -> {service_url} (hostnames: {[r.get('hostname') for r in ingress_rules]})"
            )
            return True
        except Exception as e:
            logger.error(
                f"[CloudflareTunnel] Error configuring ingress with hostname: {e}"
            )
            return False

    async def _create_network_hostname_route(
        self,
        client: httpx.AsyncClient,
        hostname: str,
        tunnel_id: str,
        service_url: str,
    ) -> bool:
        """
        Create a Zero Trust Networks hostname route so the hostname appears in
        the dashboard under Networks → Hostname routes. Optional: tunnel ingress
        (PUT configurations) already routes traffic; this may affect dashboard visibility.
        Non-fatal on failure.
        """
        url = f"{CLOUDFLARE_API_BASE}/accounts/{self.account_id}/{NETWORKS_HOSTNAME_ROUTES_PATH}"
        body = {"hostname": hostname, "tunnel_id": tunnel_id, "service": service_url}
        try:
            response = await client.post(
                url,
                headers=self.headers,
                json=body,
                timeout=15.0,
            )
            if response.status_code in (200, 201):
                data = response.json()
                if data.get("success"):
                    logger.info(
                        f"[CloudflareTunnel] ✅ Created network hostname route: {hostname}"
                    )
                    return True
            # Log so we can adjust path/body if dashboard still doesn't show hostname routes
            logger.info(
                f"[CloudflareTunnel] Create hostname route {response.status_code}: {response.text[:300]}"
            )
        except Exception as e:
            logger.debug(f"[CloudflareTunnel] Create hostname route error: {e}")
        return False

    async def ensure_tunnel_ingress(
        self,
        tunnel_id: str,
        workspace_id: str,
        local_port: Optional[int] = None,
    ) -> bool:
        """
        Public helper: re-apply ingress + hostname route for an existing workspace tunnel.
        Called on every register so the hostname is always configured even if
        provision ran before the hostname-route code was added.
        """
        hostname = self._workspace_tunnel_public_hostname(workspace_id, tunnel_id)
        try:
            async with httpx.AsyncClient() as client:
                ok = await self._configure_tunnel_ingress_public_hostname(
                    client, tunnel_id, hostname, local_port
                )
                if ok:
                    logger.info(
                        f"[CloudflareTunnel] ensure_tunnel_ingress: ingress set for {tunnel_id} hostname={hostname}"
                    )
                else:
                    logger.warning(
                        f"[CloudflareTunnel] ensure_tunnel_ingress: failed to set ingress for {tunnel_id}"
                    )
                return ok
        except Exception as e:
            logger.error(f"[CloudflareTunnel] ensure_tunnel_ingress error: {e}")
            return False

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
        When subdomain is None, only origin (catch-all) is set; when subdomain is set,
        requires CLOUDFLARE_TUNNEL_DOMAIN.
        """
        if subdomain is not None and not self.has_domain():
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
            ok, err = await self._put_tunnel_config(client, tunnel_id, config)
            if not ok:
                logger.error(f"[CloudflareTunnel] Failed to configure ingress: {err}")
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
