from typing import List

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.modules.auth.auth_service import AuthService
from app.modules.intelligence.tools.tool_schema import ToolInfo
from app.modules.intelligence.tools.tool_service import ToolService

router = APIRouter()


@router.get("/tools/list_tools", response_model=List[ToolInfo])
async def list_tools(
    db: Session = Depends(get_db),
    user=Depends(AuthService.check_auth),
):
    user_id = user["user_id"]
    tool_service = ToolService(db, user_id)
    return tool_service.list_tools()
