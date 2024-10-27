from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.modules.auth.auth_service import AuthService
from app.modules.intelligence.tools.tool_schema import (
    ToolInfo,
    ToolRequest,
    ToolResponse,
)
from app.modules.intelligence.tools.tool_service import ToolService

router = APIRouter()


@router.post("/tools/run_tool", response_model=ToolResponse)
async def run_tool(
    request: ToolRequest,
    user_id: str,
    db: Session = Depends(get_db),
    # hmac_signature: str = Header(..., alias="X-HMAC-Signature"),
):
    # if not AuthService.verify_hmac_signature(request.model_dump(), hmac_signature):
    #     raise HTTPException(status_code=401, detail="Unauthorized")

    tool_service = ToolService(db, user_id)
    try:
        result = await tool_service.run_tool(request.tool_id, request.params)
        return ToolResponse(results=result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {str(e)}"
        )


@router.get("/tools/list_tools", response_model=List[ToolInfo])
async def list_tools(
    db: Session = Depends(get_db),
    user=Depends(AuthService.check_auth),
):
    user_id = user["user_id"]
    tool_service = ToolService(db, user_id)
    return tool_service.list_tools()


@router.get("/tools/list_tools_hmac", response_model=List[ToolInfo])
async def list_tools_hmac(
    user_id: str = Query(...),
    db: Session = Depends(get_db),
    # hmac_signature: str = Header(..., alias="X-HMAC-Signature"),
):
    print("[DEBUG] Received request to list_tools_hmac endpoint")
    print(f"[DEBUG] User ID: {user_id}")

    # Uncomment and modify this section when you re-enable HMAC verification
    # print(f"[DEBUG] HMAC Signature: {hmac_signature}")
    # if not AuthService.verify_hmac_signature_for_get(user_id, hmac_signature):
    #     print(f"[DEBUG] HMAC verification failed for user_id: {user_id}")
    #     raise HTTPException(status_code=401, detail="Unauthorized")
    # print(f"[DEBUG] HMAC verification successful for user_id: {user_id}")

    print(f"[DEBUG] Creating ToolService instance for user_id: {user_id}")
    tool_service = ToolService(db, user_id)

    print("[DEBUG] Calling list_tools() method")
    tools = tool_service.list_tools()

    print(f"[DEBUG] Number of tools retrieved: {len(tools)}")
    print(f"[DEBUG] Returning tools: {tools}")

    return tools
