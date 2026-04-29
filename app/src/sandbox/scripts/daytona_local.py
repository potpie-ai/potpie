"""Mint a Daytona dev API key against the local docker-compose stack.

Local-only. Scripts the dex OIDC password login that the dashboard would
normally do for the bundled `dev@daytona.io` user, then creates an API key
scoped to the user's personal organization. Sets a default region first if
the org has none.

Usage:
    python -m scripts.daytona_local                 # prints JSON to stdout
    python -m scripts.daytona_local --env-file .env # writes shell-style env
    python -m scripts.daytona_local --check          # only verifies the stack
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import http.cookiejar
import json
import os
import re
import secrets
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

DEFAULT_DASHBOARD = os.getenv(
    "DAYTONA_DASHBOARD_URL",
    f"http://localhost:{os.getenv('DAYTONA_DASHBOARD_PORT', '3010')}",
)
DEFAULT_DEX = os.getenv("DAYTONA_DEX_URL", "http://localhost:5556/dex")
DEFAULT_EMAIL = os.getenv("DAYTONA_DEV_EMAIL", "dev@daytona.io")
DEFAULT_PASSWORD = os.getenv("DAYTONA_DEV_PASSWORD", "password")
DEFAULT_PERMISSIONS = (
    "write:sandboxes",
    "delete:sandboxes",
    "write:snapshots",
    "delete:snapshots",
    "read:runners",
)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ARG002
        raise urllib.error.HTTPError(req.full_url, code, msg, headers, fp)


def _pkce() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(64)[:64]
    challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
        .rstrip(b"=")
        .decode()
    )
    return verifier, challenge


def _login(*, dex: str, dashboard: str, email: str, password: str) -> str:
    verifier, challenge = _pkce()
    authorize = (
        f"{dex}/auth?client_id=daytona&response_type=code"
        f"&redirect_uri={urllib.parse.quote(dashboard)}"
        f"&scope=openid+profile+email+offline_access+groups"
        f"&state=potpie-setup&code_challenge={challenge}&code_challenge_method=S256"
    )
    cj = http.cookiejar.CookieJar()
    follow = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
    no_follow = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(cj), _NoRedirect()
    )

    body = follow.open(authorize).read().decode()
    m = re.search(r'<form[^>]+action="([^"]+)"', body)
    if not m:
        raise RuntimeError("dex login form not found")
    base = dex.rsplit("/dex", 1)[0]
    form_path = m.group(1).replace("&amp;", "&")
    form_url = base + form_path if form_path.startswith("/") else form_path

    next_url = form_url
    next_data: bytes | None = urllib.parse.urlencode(
        {"login": email, "password": password}
    ).encode()
    next_method = "POST"
    code: str | None = None
    for _ in range(10):
        try:
            no_follow.open(
                urllib.request.Request(next_url, data=next_data, method=next_method)
            )
        except urllib.error.HTTPError as e:
            if e.code in (301, 302, 303, 307):
                loc = e.headers.get("Location") or ""
                if not loc.startswith("http"):
                    loc = urllib.parse.urljoin(next_url, loc)
                if loc.startswith(dashboard):
                    qs = urllib.parse.urlparse(loc).query
                    code = urllib.parse.parse_qs(qs).get("code", [None])[0]
                    break
                next_url, next_data, next_method = loc, None, "GET"
                continue
            raise
        else:
            html = follow.open(next_url).read().decode()
            chosen: dict[str, str] | None = None
            for f in re.finditer(
                r'<form[^>]*method="post"[^>]*>([\s\S]*?)</form>',
                html,
                re.IGNORECASE,
            ):
                fields = {
                    n.group(1): n.group(2)
                    for n in re.finditer(
                        r'<input[^>]*name="([^"]+)"[^>]*value="([^"]*)"', f.group(1)
                    )
                }
                if fields.get("approval") == "approve":
                    chosen = fields
                    break
            if not chosen:
                raise RuntimeError("dex approval form not found")
            next_data = urllib.parse.urlencode(chosen).encode()
            next_method = "POST"
    if not code:
        raise RuntimeError("dex did not return an authorization code")

    token_resp = urllib.request.urlopen(
        urllib.request.Request(
            f"{dex}/token",
            data=urllib.parse.urlencode(
                {
                    "grant_type": "authorization_code",
                    "code": code,
                    "client_id": "daytona",
                    "redirect_uri": dashboard,
                    "code_verifier": verifier,
                }
            ).encode(),
        )
    )
    return json.loads(token_resp.read())["id_token"]


def mint_dev_api_key(
    *,
    dashboard: str = DEFAULT_DASHBOARD,
    dex: str = DEFAULT_DEX,
    email: str = DEFAULT_EMAIL,
    password: str = DEFAULT_PASSWORD,
    name: str | None = None,
    permissions: tuple[str, ...] = DEFAULT_PERMISSIONS,
) -> tuple[str, str]:
    """Return ``(api_key, organization_id)`` for the local dev user."""
    jwt = _login(dex=dex, dashboard=dashboard, email=email, password=password)
    api = f"{dashboard}/api"
    auth_hdr = {"Authorization": f"Bearer {jwt}"}
    orgs = json.loads(
        urllib.request.urlopen(
            urllib.request.Request(f"{api}/organizations", headers=auth_hdr)
        ).read()
    )
    if not orgs:
        raise RuntimeError("dev user has no organizations")
    org = orgs[0]
    org_id = org["id"]

    if not org.get("defaultRegionId"):
        regions = json.loads(
            urllib.request.urlopen(
                urllib.request.Request(
                    f"{api}/regions",
                    headers={**auth_hdr, "X-Daytona-Organization-ID": org_id},
                )
            ).read()
        )
        if regions:
            try:
                urllib.request.urlopen(
                    urllib.request.Request(
                        f"{api}/organizations/{org_id}/default-region",
                        data=json.dumps({"defaultRegionId": regions[0]["id"]}).encode(),
                        method="PATCH",
                        headers={
                            **auth_hdr,
                            "Content-Type": "application/json",
                            "X-Daytona-Organization-ID": org_id,
                        },
                    )
                )
            except urllib.error.HTTPError as e:
                if e.code != 409:  # already set
                    raise

    key_name = name or f"potpie-{secrets.token_hex(4)}"
    body = json.dumps({"name": key_name, "permissions": list(permissions)}).encode()
    resp = urllib.request.urlopen(
        urllib.request.Request(
            f"{api}/api-keys",
            data=body,
            headers={
                **auth_hdr,
                "Content-Type": "application/json",
                "X-Daytona-Organization-ID": org_id,
            },
        )
    )
    payload = json.loads(resp.read())
    api_key = payload.get("value") or payload.get("key") or payload["apiKey"]
    return api_key, org_id


def _stack_reachable(dashboard: str) -> bool:
    try:
        with urllib.request.urlopen(f"{dashboard}/api/health", timeout=3) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dashboard", default=DEFAULT_DASHBOARD)
    parser.add_argument("--dex", default=DEFAULT_DEX)
    parser.add_argument("--email", default=DEFAULT_EMAIL)
    parser.add_argument("--password", default=DEFAULT_PASSWORD)
    parser.add_argument("--name", default=None, help="API key name (default: random)")
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Write SANDBOX_* env vars to this file in addition to printing JSON.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only verify the stack is reachable; do not mint a key.",
    )
    args = parser.parse_args()

    if not _stack_reachable(args.dashboard):
        sys.stderr.write(
            f"Daytona dashboard not reachable at {args.dashboard} — start the "
            "stack with scripts/setup-daytona-local.sh first.\n"
        )
        return 2
    if args.check:
        print(json.dumps({"reachable": True, "dashboard": args.dashboard}))
        return 0

    api_key, org_id = mint_dev_api_key(
        dashboard=args.dashboard,
        dex=args.dex,
        email=args.email,
        password=args.password,
        name=args.name,
    )
    api_url = f"{args.dashboard}/api"
    payload = {
        "api_url": api_url,
        "api_key": api_key,
        "organization_id": org_id,
        "dashboard": args.dashboard,
    }
    print(json.dumps(payload, indent=2))

    if args.env_file:
        args.env_file.write_text(
            "# Generated by scripts/daytona_local.py — do not commit.\n"
            f"DAYTONA_API_URL={api_url}\n"
            f"DAYTONA_API_KEY={api_key}\n"
            f"DAYTONA_DASHBOARD_URL={args.dashboard}\n"
            f"DAYTONA_ORGANIZATION_ID={org_id}\n"
            "DAYTONA_SNAPSHOT=potpie/agent-sandbox:0.1.0\n"
            "SANDBOX_WORKSPACE_PROVIDER=daytona\n"
            "SANDBOX_RUNTIME_PROVIDER=daytona\n",
            encoding="utf-8",
        )
        sys.stderr.write(f"wrote env to {args.env_file}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
