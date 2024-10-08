
from fastapi import Depends, HTTPException, APIRouter, Header
from app.core.database import get_db
from app.modules.auth.auth_service import AuthService
from app.modules.intelligence.tools.tool_schema import ToolRequest, ToolResponse
from app.modules.intelligence.tools.tool_service import ToolService
from sqlalchemy.orm import Session
from typing import List
from app.modules.intelligence.tools.tool_schema import ToolInfo


router = APIRouter()

@router.post("/tools/run_tool", response_model=ToolResponse)
async def run_tool(
    request: ToolRequest,
    db: Session = Depends(get_db),
    user=Depends(AuthService.check_auth),
    hmac_signature: str = Header(..., alias="X-HMAC-Signature"),
):
    user_id = user["user_id"]
    if not AuthService.verify_hmac_signature(request.model_dump(), hmac_signature):
        raise HTTPException(status_code=401, detail="Unauthorized")

    tool_service = ToolService(db, user_id)
    try:
        result = await tool_service.run_tool(request.tool_id, request.params)
        return ToolResponse(results=result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@router.get("/tools/list_tools", response_model=List[ToolInfo])
async def list_tools(
    db: Session = Depends(get_db),
    user=Depends(AuthService.check_auth),
):
    user_id = user["user_id"]
    tool_service = ToolService(db, user_id)
    return tool_service.list_tools()