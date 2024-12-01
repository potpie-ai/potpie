from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.modules.auth.auth_service import AuthService
from app.modules.usage.usage_controller import UsageController
from app.modules.usage.usage_schema import UsageResponse

router = APIRouter()

class UsageAPI:
    @staticmethod
    @router.get("/usage", response_model=UsageResponse)
    def read_usage(
        db: Session = Depends(get_db),
        current_user: dict = Depends(AuthService.check_auth),
    ):
        user_id = current_user["user_id"]
        
        controller = UsageController(db)
        usage = controller.get_usage_by_user_id(user_id)
        if usage is None:
            raise HTTPException(status_code=404, detail="Usage not found")
        return usage

    @staticmethod
    @router.post("/usage/create_usage", response_model=UsageResponse)
    def create_usage(
        db: Session = Depends(get_db),
        current_user: dict = Depends(AuthService.check_auth),
    ):
        user_id = current_user["user_id"]
        
        controller = UsageController(db)
        usage = controller.create_usage(user_id)
        return usage