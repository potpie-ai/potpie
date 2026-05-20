import logging
import os

import httpx
from dotenv import load_dotenv
from fastapi import Depends, HTTPException, Request, Response, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from firebase_admin import auth
from firebase_admin.exceptions import FirebaseError

load_dotenv(override=True)

# Timeout for Firebase Identity Toolkit (default 30s read, 10s connect)
LOGIN_TIMEOUT = httpx.Timeout(30.0, connect=10.0, read=30.0)


class AuthService:
    def login(self, email, password):
        """Sync login (e.g. for tests)."""
        log_prefix = "AuthService::login:"
        identity_tool_kit_id = os.getenv("GOOGLE_IDENTITY_TOOL_KIT_KEY")
        identity_url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={identity_tool_kit_id}"

        with httpx.Client(timeout=LOGIN_TIMEOUT) as client:
            try:
                user_auth_response = client.post(
                    url=identity_url,
                    json={
                        "email": email,
                        "password": password,
                        "returnSecureToken": True,
                    },
                )
                user_auth_response.raise_for_status()
                return user_auth_response.json()
            except httpx.HTTPStatusError as e:
                logging.warning(
                    "%s upstream auth failed: status=%s", log_prefix, e.response.status_code
                )
                try:
                    detail = e.response.json()
                except Exception:
                    detail = e.response.text or "Upstream auth error"
                raise HTTPException(
                    status_code=e.response.status_code,
                    detail=detail,
                )
            except httpx.HTTPError as e:
                logging.warning("%s upstream auth failed: httpx error", log_prefix)
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail="Upstream auth request failed",
                ) from e

    async def login_async(self, email, password):
        """Non-blocking login using httpx. Use from FastAPI routes."""
        log_prefix = "AuthService::login:"
        identity_tool_kit_id = os.getenv("GOOGLE_IDENTITY_TOOL_KIT_KEY")
        identity_url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={identity_tool_kit_id}"
        response = None
        try:
            async with httpx.AsyncClient(timeout=LOGIN_TIMEOUT) as client:
                response = await client.post(
                    identity_url,
                    json={
                        "email": email,
                        "password": password,
                        "returnSecureToken": True,
                    },
                )
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            logging.warning(
                "%s upstream auth failed: status=%s", log_prefix, e.response.status_code
            )
            try:
                detail = e.response.json()
            except Exception:
                detail = e.response.text or "Upstream auth error"
            raise HTTPException(
                status_code=e.response.status_code,
                detail=detail,
            )
        except httpx.HTTPError as e:
            logging.warning("%s upstream auth failed: httpx error", log_prefix)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Upstream auth request failed",
            ) from e

    def signup(self, email: str, password: str, name: str) -> tuple:
        try:
            user = auth.create_user(email=email, password=password, display_name=name)
            return {"user": user, "message": "New user created successfully"}, None
        except FirebaseError as fe:
            return None, {"error": f"Firebase error: {fe.message}"}
        except ValueError as _ve:
            return None, {"error": "Invalid input data provided."}
        except Exception as e:
            return None, {"error": f"An unexpected error occurred: {str(e)}"}

    @staticmethod
    def create_custom_token(
        uid: str, additional_claims: dict | None = None
    ) -> str | None:
        """
        Create a Firebase custom token for the given uid (e.g. for VS Code extension).

        F-14: callers should pass `additional_claims={"surface": "vscode-ext"}`
        so the token is scoped — if it leaks, downstream code can refuse it on
        non-extension surfaces. Returns the token string or None on error.
        """
        try:
            token = auth.create_custom_token(uid, additional_claims)
            return token.decode("utf-8") if isinstance(token, bytes) else token
        except Exception as e:
            logging.warning("create_custom_token failed for uid=%s: %s", uid, e)
            return None

    @staticmethod
    async def check_auth(
        request: Request,
        res: Response,
        credential: HTTPAuthorizationCredentials = Depends(
            HTTPBearer(auto_error=False)
        ),
    ):
        # When invoked manually (e.g. auth_handler.check_auth(request, None)), the
        # third argument defaults to the Depends() object. Resolve Bearer token from
        # the request so both dependency-injected and manual calls work.
        if not isinstance(credential, HTTPAuthorizationCredentials):
            auth_header = request.headers.get("Authorization")
            if auth_header and auth_header.startswith("Bearer "):
                credential = HTTPAuthorizationCredentials(
                    scheme="Bearer", credentials=auth_header[7:].strip()
                )
            else:
                credential = None

        logging.info("DEBUG: AuthService.check_auth called")
        logging.info("DEBUG: Credential provided: %s", credential is not None)

        # F-11: dev-mode bypass requires BOTH isDevelopmentMode=enabled and
        # POTPIE_ALLOW_DEV_AUTH=1 (fail-closed second gate).
        from app.modules.auth.api_key_deps import dev_auth_enabled

        if dev_auth_enabled() and credential is None:
            request.state.user = {"user_id": os.getenv("defaultUsername")}
            logging.info("DEBUG: Development mode enabled. Using Mock Authentication.")
            return {
                "user_id": os.getenv("defaultUsername"),
                "email": "defaultuser@potpie.ai",
            }
        else:
            if credential is None:
                logging.error(
                    "DEBUG: No credential provided and not in development mode"
                )
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Bearer authentication is needed",
                    headers={"WWW-Authenticate": 'Bearer realm="auth_required"'},
                )
            try:
                logging.info(
                    "DEBUG: Verifying Firebase token: %s...",
                    credential.credentials[:20],
                )
                # F-13: honor Firebase token revocation (logout-all, password
                # change, account disabled). Cost is one extra RPC per request
                # against Firebase's revocation list; acceptable for the
                # security benefit since this dependency gates every auth'd
                # API route.
                decoded_token = auth.verify_id_token(
                    credential.credentials, check_revoked=True
                )
                # Normalize token to always include "user_id" for consistency across environments
                # Firebase tokens use "uid", but our codebase expects "user_id"
                if "uid" in decoded_token and "user_id" not in decoded_token:
                    decoded_token["user_id"] = decoded_token["uid"]
                logging.info(
                    "DEBUG: Successfully verified token for user: %s",
                    decoded_token.get("user_id", decoded_token.get("uid", "unknown")),
                )
                logging.info(
                    "DEBUG: Token email: %s",
                    decoded_token.get("email", "unknown"),
                )
                request.state.user = decoded_token
            except Exception as err:
                # F-12: do not leak Firebase exception text to unauthenticated
                # callers (aud/iss mismatches, project IDs etc.). Log full
                # context server-side and return a static detail.
                logging.warning(
                    "AuthService.check_auth: Firebase token verification failed: %s",
                    err,
                )
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token",
                    headers={"WWW-Authenticate": 'Bearer error="invalid_token"'},
                )
            if res is not None:
                res.headers["WWW-Authenticate"] = 'Bearer realm="auth_required"'
            return decoded_token


auth_handler = AuthService()
