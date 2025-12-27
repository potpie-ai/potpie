from fastapi import Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.modules.auth.auth_service import AuthService
from app.modules.users.user_controller import UserController
from app.modules.users.user_schema import (
    UserProfileResponse,
    OnboardingDataRequest,
    OnboardingDataResponse,
)
from app.modules.utils.APIRouter import APIRouter
from app.modules.utils.logger import setup_logger

router = APIRouter()
logger = setup_logger(__name__)


class UserAPI:
    @router.get("/user/{user_id}/public-profile", response_model=UserProfileResponse)
    async def fetch_user_profile_pic(
        user_id: str,
        user=Depends(AuthService.check_auth),
        db: Session = Depends(get_db),
    ):
        controller = UserController(db)
        return await controller.get_user_profile_pic(user_id)

    @router.post("/user/onboarding", response_model=OnboardingDataResponse)
    async def save_onboarding_data(
        request: OnboardingDataRequest,
        user=Depends(AuthService.check_auth),
        db: Session = Depends(get_db),
    ):
        """
        Save onboarding data to Firestore using Firebase Admin SDK.
        This endpoint uses admin privileges to write to Firestore,
        bypassing client-side permission issues.
        """
        try:
            # Get the authenticated user's UID from the token
            # Firebase tokens use 'uid' as the key, but we also check 'user_id' for compatibility
            authenticated_uid = user.get("uid") or user.get("user_id")

            # Verify the authenticated user matches the request UID
            if authenticated_uid != request.uid:
                logger.warning(
                    f"UID mismatch: authenticated={authenticated_uid}, requested={request.uid}"
                )
                raise HTTPException(
                    status_code=403,
                    detail="You can only save onboarding data for your own account",
                )

            # Import firestore here to avoid circular imports
            from firebase_admin import firestore
            from datetime import datetime, timezone

            # Get Firestore client
            db_firestore = firestore.client()

            # Prepare the document data
            user_doc = {
                "uid": request.uid,
                "email": request.email,
                "name": request.name,
                "source": request.source,
                "industry": request.industry,
                "jobTitle": request.jobTitle,
                "companyName": request.companyName,
                "signedUpAt": datetime.now(timezone.utc).isoformat(),
            }

            # Save to Firestore using admin SDK (has full permissions)
            doc_ref = db_firestore.collection("users").document(request.uid)
            doc_ref.set(user_doc, merge=True)

            logger.info(f"Successfully saved onboarding data for user {request.uid}")

            return OnboardingDataResponse(
                success=True, message="Onboarding information saved successfully"
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error saving onboarding data: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500, detail=f"Failed to save onboarding data: {str(e)}"
            )
