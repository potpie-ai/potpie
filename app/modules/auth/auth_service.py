import logging
import os

import requests
from dotenv import load_dotenv
from fastapi import Depends, HTTPException, Request, Response, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from firebase_admin import auth

load_dotenv(override=True)


class AuthService:
    def login(self, email, password):
        log_prefix = "AuthService::login:"
        identity_tool_kit_id = os.getenv("GOOGLE_IDENTITY_TOOL_KIT_KEY")
        identity_url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={identity_tool_kit_id}"

        user_auth_response = requests.post(
            url=identity_url,
            json={
                "email": email,
                "password": password,
                "returnSecureToken": True,
            },
        )

        try:
            user_auth_response.raise_for_status()
            return user_auth_response.json()
        except Exception as e:
            logging.exception(f"{log_prefix} {str(e)}")
            raise Exception(user_auth_response.json())

    def signup(self, email, password, name):
        user = auth.create_user(email=email, password=password, display_name=name)
        return user

    @classmethod
    @staticmethod
    async def check_auth(
        request: Request,
        res: Response,
        credential: HTTPAuthorizationCredentials = Depends(
            HTTPBearer(auto_error=False)
        ),
    ):
        # Check if the application is in debug mode
        if os.getenv("isDevelopmentMode") == "enabled" and credential is None:
            request.state.user = {"user_id": os.getenv("defaultUsername")}
            logging.info("Development mode enabled. Using Mock Authentication.")
            return {
                "user_id": os.getenv("defaultUsername"),
                "email": "defaultuser@potpie.ai",
            }
        else:
            if credential is None:
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


auth_handler = AuthService()
