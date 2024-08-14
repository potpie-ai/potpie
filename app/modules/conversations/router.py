import json
from typing import List
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from sqlalchemy.orm import Session
from app.core.database import get_db
import asyncio

from app.modules.conversations.conversation.controller import ConversationController

from .conversation.schema import (
    CreateConversationRequest, 
    CreateConversationResponse, 
    ConversationResponse, 
    ConversationInfoResponse, 
)

from .message.schema import (
    MessageRequest,
    MessageResponse
)

router = APIRouter()

class ConversationAPI:

    @staticmethod
    def get_controller():
        return ConversationController()
    

    @staticmethod
    @router.post("/conversations/", response_model=CreateConversationResponse)
    async def create_conversation(
        conversation: CreateConversationRequest,
        controller: ConversationController = Depends(get_controller),
        db: Session = Depends(get_db),
    ):
        return await controller.create_conversation(conversation)

    @staticmethod
    @router.get("/conversations/{conversation_id}/", response_model=ConversationResponse)
    async def get_conversation(
        conversation_id: str,
        controller: ConversationController = Depends(get_controller),
        db: Session = Depends(get_db)
    ):
        return await controller.get_conversation(conversation_id)

    @staticmethod
    @router.get("/conversations/{conversation_id}/info/", response_model=ConversationInfoResponse)
    async def get_conversation_info(
        conversation_id: str,
        controller: ConversationController = Depends(get_controller),
        db: Session = Depends(get_db)
    ):
        return await controller.get_conversation_info(conversation_id)

    @staticmethod
    @router.get("/conversations/{conversation_id}/messages/", response_model=List[MessageResponse])
    async def get_conversation_messages(
        conversation_id: str,
        start: int = Query(0, ge=0),  # Start index, default is 0
        limit: int = Query(10, ge=1),  # Number of items to return, default is 10
        controller: ConversationController = Depends(get_controller),
        db: Session = Depends(get_db)
    ):
        return await controller.get_conversation_messages(conversation_id, start, limit)

    @staticmethod
    @router.post("/conversations/{conversation_id}/message/")
    async def post_message(
        conversation_id: str,
        message: MessageRequest,
        controller: ConversationController = Depends(get_controller),
        db: Session = Depends(get_db)
    ):  
        return await controller.post_message(conversation_id, message, db, user_id='abc')

    @staticmethod
    @router.post("/conversations/{conversation_id}/regenerate/", response_model=MessageResponse)
    async def regenerate_last_message(
        conversation_id: str,
        controller: ConversationController = Depends(get_controller),
        db: Session = Depends(get_db)
    ):
        return await controller.regenerate_last_message(conversation_id)

    @staticmethod
    @router.delete("/conversations/{conversation_id}/", response_model=dict)
    async def delete_conversation(
        conversation_id: str, 
        controller: ConversationController = Depends(get_controller),
        db: Session = Depends(get_db)
    ):
        return await controller.delete_conversation(conversation_id)
    
    @staticmethod
    @router.post("/conversations/{conversation_id}/stop/", response_model=dict)
    async def stop_generation(
        conversation_id: str,
        controller: ConversationController = Depends(get_controller),
        db: Session = Depends(get_db)
    ):
        return await controller.stop_generation(conversation_id)