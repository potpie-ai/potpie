# app/core/base_store.py

from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import AsyncSession

class BaseStore:
    """
    A base class for all data stores that holds the database sessions
    needed during the sync-to-async migration period.
    """
    def __init__(self, db: Session, async_db: AsyncSession):
        self.db = db  # For legacy sync dependencies
        self.async_db = async_db  # For new async queries