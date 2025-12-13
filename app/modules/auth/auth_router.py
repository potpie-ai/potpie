import json
import os
from datetime import datetime
from collections import defaultdict
from time import time
import ipaddress

import requests
from dotenv import load_dotenv
from fastapi import Depends, Request
from fastapi.responses import JSONResponse, Response
from fastapi.exceptions import HTTPException

from sqlalchemy.orm import Session

from app.core.database import get_db
from app.modules.auth.auth_schema import LoginRequest
from app.modules.auth.auth_service import auth_handler
from app.modules.users.user_schema import CreateUser
from app.modules.users.user_service import UserService
from app.modules.utils.APIRouter import APIRouter
from app.modules.utils.posthog_helper import PostHogClient

SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", None)

auth_router = APIRouter()
load_dotenv(override=True)


# Rate limiting configuration
MAX_REQUESTS_PER_MINUTE = 5  # 5 requests per minute per IP
RATE_LIMIT_WINDOW = 60  # seconds

# Store request timestamps per IP: { ip: [timestamp, ...] }
request_tracker = defaultdict(list)


def get_client_ip(request: Request) -> str:
    """
    Extract client IP from request, handling common proxy headers.
    Returns an empty string if a reliable client IP cannot be determined.
    """
    # Check X-Forwarded-For first (may contain a comma-separated list)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # Take the left-most (original client) value
        candidate = forwarded_for.split(",")[0].strip()
        try:
            ipaddress.ip_address(candidate)
            return candidate
        except ValueError:
            # invalid IP in header; fall through to other checks
            pass

    # Check X-Real-IP
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        try:
            ipaddress.ip_address(real_ip.strip())
            return real_ip.strip()
        except ValueError:
            pass

    # Fallback to the transport-level peer IP if available
    if hasattr(request, "client") and request.client:
        try:
            ipaddress.ip_address(request.client.host)
            return request.client.host
        except Exception:
            pass

    # If we truly can't determine the IP, return empty string so callers can handle it
    return ""


def check_rate_limit(request: Request) -> bool:
    """
    Check if client has exceeded rate limit.
    Returns True if request is allowed, False if rate limited or client IP can't be determined.
    Implementation: sliding window using timestamps per IP.
    """
    client_ip = get_client_ip(request)
    if not client_ip:
        # Safer to reject when the client IP cannot be determined reliably
        return False

    now = time()
    window_start = now - RATE_LIMIT_WINDOW

    # Prune old timestamps outside the window
    timestamps = request_tracker[client_ip]
    timestamps = [ts for ts in timestamps if ts >= window_start]
    request_tracker[client_ip] = timestamps

    # Count requests in window
    if len(timestamps) >= MAX_REQUESTS_PER_MINUTE:
        return False  # Rate limited

    # Record this request
    request_tracker[client_ip].append(now)
    return True  # Request allowed


async def send_slack_message(message: str):
    payload = {"text": message}
    if SLACK_WEBHOOK_URL:
        requests.post(SLACK_WEBHOOK_URL, json=payload)


class AuthAPI:
    @auth_router.post("/login")
    async def login(login_request: LoginRequest, request: Request):
        # Check rate limit
        if not check_rate_limit(request):
            return JSONResponse(
                content={"error": f"Rate limit exceeded. Maximum {MAX_REQUESTS_PER_MINUTE} requests per minute allowed."},
                status_code=429,
            )
        
        email, password = login_request.email, login_request.password

        try:
            res = auth_handler.login(email=email, password=password)
            id_token = res.get("idToken")
            return JSONResponse(content={"token": id_token}, status_code=200)
        except ValueError:
            return JSONResponse(
                content={"error": "Invalid email or password"}, status_code=401
            )
        except HTTPException as he:
            return JSONResponse(
                content={"error": f"HTTP Error: {str(he)}"}, status_code=he.status_code
            )
        except Exception as e:
            return JSONResponse(content={"error": f"ERROR: {str(e)}"}, status_code=400)

    @auth_router.post("/signup")
    async def signup(request: Request, db: Session = Depends(get_db)):
        # Check rate limit
        if not check_rate_limit(request):
            return JSONResponse(
                content={"error": f"Rate limit exceeded. Maximum {MAX_REQUESTS_PER_MINUTE} requests per minute allowed."},
                status_code=429,
            )
        
        body = json.loads(await request.body())
        uid = body["uid"]
        oauth_token = body["accessToken"]
        user_service = UserService(db)
        user = user_service.get_user_by_uid(uid)
        if user:
            message, error = user_service.update_last_login(uid, oauth_token)
            if error:
                return Response(content=message, status_code=400)
            else:
                return Response(
                    content=json.dumps({"uid": uid, "exists": True}),
                    status_code=200,
                )
        else:
            first_login = datetime.utcnow()
            provider_info = body["providerData"][0]
            provider_info["access_token"] = oauth_token
            user = CreateUser(
                uid=uid,
                email=body["email"],
                display_name=body["displayName"],
                email_verified=body["emailVerified"],
                created_at=first_login,
                last_login_at=first_login,
                provider_info=provider_info,
                provider_username=body["providerUsername"],
            )
            uid, message, error = user_service.create_user(user)

            await send_slack_message(
                f"New signup: {body['email']} ({body['displayName']})"
            )

            PostHogClient().send_event(
                uid,
                "signup_event",
                {
                    "email": body["email"],
                    "display_name": body["displayName"],
                    "github_username": body["providerUsername"],
                },
            )
            if error:
                return Response(content=message, status_code=400)
            return Response(
                content=json.dumps({"uid": uid, "exists": False}),
                status_code=201,
            )
