import hashlib
import hmac
import json
import logging
import os
from typing import Union, Callable

import requests
from dotenv import load_dotenv
from fastapi import HTTPException, Request, Response, status

from firebase_admin import auth
from firebase_admin.auth import UserRecord
from abc import ABC, abstractmethod
from app.modules.auth.auth_schema import User
from starlette.types import ASGIApp

load_dotenv(override=True)


class AuthService(ABC):
    """Inteface for authentication service, auth service will be implemented by different providers."""

    @abstractmethod
    def login(self, email: str, password: str) -> str:
        pass

    @abstractmethod
    def signup(self, email: str, password: str, name: str):
        pass

    @abstractmethod
    async def authenticate(self, token: str) -> User:
        pass


def get_auth_middleware(auth_service: AuthService):
    async def authenticate(
        request: Request,
        call_next: Callable[[Request], Response],
    ):
        credential = request.headers.get("Authorization")
        if not credential:
            raise HTTPException(
                status_code=401, detail="Bearer authentication is needed"
            )
        try:
            token = credential.split("Bearer ")[1]
            user = await auth_service.authenticate(token)
            request.state.user = user
        except Exception as err:
            raise HTTPException(status_code=401, detail="Invalid token")

        response = await call_next(request)
        return response

    return authenticate


class GoogleIdentityAuthService(AuthService):

    def __init__(self, identity_toolkit_key: str) -> None:
        self.identity_url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={identity_toolkit_key}"

    def login(self, email: str, password: str) -> str:
        log_prefix = "AuthService:GoogleIdentity::login"

        user_auth_response = requests.post(
            url=self.identity_url,
            json={
                "email": email,
                "password": password,
                "returnSecureToken": True,
            },
        )

        try:
            user_auth_response.raise_for_status()
            return user_auth_response.json().get("idToken")
        except Exception as e:
            logging.exception(f"{log_prefix} {str(e)}")
            raise Exception(user_auth_response.json())

    def signup(self, email: str, password: str, name: str) -> UserRecord:
        user = auth.create_user(email=email, password=password, display_name=name)
        return user

    async def authenticate(
        request: Request,
        res: Response,
        token: str,
    ):
        # Check if the application is in debug mode
        if os.getenv("isDevelopmentMode") == "enabled" and token is None:
            request.state.user = {"user_id": os.getenv("defaultUsername")}
            logging.info("Development mode enabled. Using Mock Authentication.")
            return {
                "user_id": os.getenv("defaultUsername"),
                "email": "defaultuser@potpie.ai",
            }
        else:
            if token is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Bearer authentication is needed",
                    headers={"WWW-Authenticate": 'Bearer realm="auth_required"'},
                )
            try:
                decoded_token = auth.verify_id_token(credential.credentials)
                request.state.user = decoded_token
            except Exception as err:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=f"Invalid authentication from Firebase. {err}",
                    headers={"WWW-Authenticate": 'Bearer error="invalid_token"'},
                )
            res.headers["WWW-Authenticate"] = 'Bearer realm="auth_required"'
            return decoded_token


class HMACService:

    def __init__(self, hmac_key: str) -> None:
        self.hmac_key = hmac_key

    def generate_hmac_signature(self, message: str) -> str:
        """Generate HMAC signature for a message string"""
        hmac_key = self.get_hmac_secret_key()
        if not hmac_key:
            raise ValueError("HMAC secret key not configured")
        hmac_obj = hmac.new(
            key=hmac_key, msg=message.encode("utf-8"), digestmod=hashlib.sha256
        )
        return hmac_obj.hexdigest()

    def verify_hmac_signature(
        self, payload_body: Union[str, dict], hmac_signature: str
    ) -> bool:
        """Verify HMAC signature matches the payload"""
        hmac_key = self.get_hmac_secret_key()
        if not hmac_key:
            raise ValueError("HMAC secret key not configured")
        payload_str = (
            payload_body
            if isinstance(payload_body, str)
            else json.dumps(payload_body, sort_keys=True)
        )
        expected_signature = hmac.new(
            key=hmac_key, msg=payload_str.encode("utf-8"), digestmod=hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(hmac_signature, expected_signature)

    def get_hmac_secret_key(self) -> bytes:
        """Get HMAC secret key from environment"""

        if not self.hmac_key:
            return b""
        return self.hmac_key.encode("utf-8")


# Use mock if GOOGLE_IDENTITY_TOOLKIT_KEY is not set or in dev mode
auth_handler = GoogleIdentityAuthService(os.getenv("GOOGLE_IDENTITY_TOOLKIT_KEY") or "")
