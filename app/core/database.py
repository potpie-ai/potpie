import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

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

# Base class for all ORM models
Base = declarative_base()


# Dependency to be used in routes
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
