import logging
import os

import requests
from dotenv import load_dotenv
from fastapi import Depends, HTTPException, Request, Response, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from firebase_admin import auth
from firebase_admin.exceptions import FirebaseError

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

    @classmethod
    @staticmethod
    async def check_auth(
        request: Request,
        res: Response,
        credential: HTTPAuthorizationCredentials = Depends(
            HTTPBearer(auto_error=False)
        ),
    ):
        logging.info("DEBUG: AuthService.check_auth called")
        logging.info(f"DEBUG: Development mode: {os.getenv('isDevelopmentMode')}")
        logging.info(f"DEBUG: Credential provided: {credential is not None}")

        # Check if the application is in debug mode
        if os.getenv("isDevelopmentMode") == "enabled" and credential is None:
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
                    f"DEBUG: Verifying Firebase token: {credential.credentials[:20]}..."
                )
                decoded_token = auth.verify_id_token(credential.credentials)
                logging.info(
                    f"DEBUG: Successfully verified token for user: {decoded_token.get('user_id', 'unknown')}"
                )
                logging.info(
                    f"DEBUG: Token email: {decoded_token.get('email', 'unknown')}"
                )
                request.state.user = decoded_token
            except Exception as err:
                logging.error(f"DEBUG: Firebase token verification failed: {str(err)}")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=f"Invalid authentication from Firebase. {err}",
                    headers={"WWW-Authenticate": 'Bearer error="invalid_token"'},
                )
            res.headers["WWW-Authenticate"] = 'Bearer realm="auth_required"'
            return decoded_token


auth_handler = AuthService()
