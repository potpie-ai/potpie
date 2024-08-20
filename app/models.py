from sqlalchemy.orm import relationship
from app.modules.conversations.message.message_model import Message
from app.modules.users.user_model import User
from app.modules.projects.projects_model import Project
from app.modules.conversations.conversation.conversation_model import Conversation

# User relationships
User.projects = relationship("Project", back_populates="user")
User.conversations = relationship("Conversation", back_populates="user")

# Project relationships
Project.user = relationship("User", back_populates="projects")

# Conversation relationships
Conversation.user = relationship("User", back_populates="conversations")
Conversation.messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")

# Message relationships
Message.conversation = relationship("Conversation", back_populates="messages")
