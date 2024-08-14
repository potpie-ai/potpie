# models.py

from sqlalchemy.orm import relationship
from app.modules.users.model import User
from app.modules.projects.model import Project
from app.modules.conversations.conversation.model import Conversation
from app.modules.conversations.message.model import Message

# Set up relationships between models
User.projects = relationship("Project", back_populates="user")
User.conversations = relationship("Conversation", back_populates="user")

Project.user = relationship("User", back_populates="projects")

Conversation.user = relationship("User", back_populates="conversations")
Conversation.messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")

Message.conversation = relationship("Conversation", back_populates="messages")
