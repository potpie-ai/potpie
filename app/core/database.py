import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession

load_dotenv(override=True)

# Create engine with connection pooling and best practices
engine = create_engine(
    os.getenv("POSTGRES_SERVER"),
    pool_size=10,  # Initial number of connections in the pool
    max_overflow=10,  # Maximum number of connections beyond pool_size
    pool_timeout=30,  # Timeout in seconds for getting a connection from the pool
    pool_recycle=1800,  # Recycle connections every 30 minutes (to avoid stale connections)
    pool_pre_ping=True,  # Check the connection is alive before using it
    echo=False,  # Set to True for SQL query logging, False in production
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Async engine
ASYNC_DATABASE_URL = os.getenv("POSTGRES_SERVER").replace(
    "postgresql://", "postgresql+asyncpg://"
)

async_engine = create_async_engine(
    ASYNC_DATABASE_URL,
    pool_size=10,
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=1800,
    pool_pre_ping=True,
    echo=False,
)

AsyncSessionLocal = sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False, # Good practice for async sessions
)

# Base class for all ORM models
Base = declarative_base()


# Dependency to be used in routes
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
        
# Dependency to be used in asynchronous routes
async def get_async_db():
    async with AsyncSessionLocal() as db:
        try:
            yield db
        finally:
            # The 'async with' context manager handles closing automatically.
            # You could also do 'await db.close()' in a more manual setup.
            pass   
