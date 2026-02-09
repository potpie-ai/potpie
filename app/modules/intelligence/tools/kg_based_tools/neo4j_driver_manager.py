import atexit
import logging
import threading
from typing import Optional

from neo4j import GraphDatabase, Driver

from app.core.config_provider import config_provider

logger = logging.getLogger(__name__)


class Neo4jDriverManager:
    """
    Thread-safe singleton manager for Neo4j driver connections.
    
    Uses lazy initialization and configures connection pooling
    for optimal resource usage.
    
    Usage:
        driver = Neo4jDriverManager.get_driver()
        with driver.session() as session:
            result = session.run(query, params)
    """
    
    _instance: Optional["Neo4jDriverManager"] = None
    _lock: threading.Lock = threading.Lock()
    _driver: Optional[Driver] = None
    
    MAX_CONNECTION_LIFETIME = 3600
    MAX_CONNECTION_POOL_SIZE = 50
    CONNECTION_ACQUISITION_TIMEOUT = 60
    
    def __new__(cls) -> "Neo4jDriverManager":
        """Ensure only one instance exists (singleton pattern)."""
        if cls._instance is None:
            with cls._lock:
                # Double-check locking pattern
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def get_driver(cls) -> Driver:
        """
        Get the shared Neo4j driver instance.
        
        Creates the driver on first call (lazy initialization).
        Thread-safe for concurrent access.
        
        Returns:
            Configured Neo4j driver with connection pooling
            
        Raises:
            ValueError: If Neo4j configuration is incomplete
            Exception: If connection to Neo4j fails
        """
        if cls._driver is None:
            with cls._lock:
                # Double-check after acquiring lock
                if cls._driver is None:
                    cls._driver = cls._create_driver()
        return cls._driver
    
    @classmethod
    def _create_driver(cls) -> Driver:
        """
        Create and configure the Neo4j driver.
        
        Configures connection pooling to prevent resource exhaustion
        under high load.
        """
        neo4j_config = config_provider.get_neo4j_config()
        
        uri = neo4j_config.get("uri")
        username = neo4j_config.get("username")
        password = neo4j_config.get("password")
        
        if not uri or not username or not password:
            raise ValueError(
                "Neo4j configuration is incomplete. "
                "Required: NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD"
            )
        
        logger.info(
            f"Creating shared Neo4j driver with pool size {cls.MAX_CONNECTION_POOL_SIZE}"
        )
        
        driver = GraphDatabase.driver(
            uri,
            auth=(username, password),
            max_connection_lifetime=cls.MAX_CONNECTION_LIFETIME,
            max_connection_pool_size=cls.MAX_CONNECTION_POOL_SIZE,
            connection_acquisition_timeout=cls.CONNECTION_ACQUISITION_TIMEOUT,
        )
        
        try:
            driver.verify_connectivity()
            logger.info("Neo4j driver connected and verified successfully")
        except Exception as e:
            logger.error(f"Failed to verify Neo4j connectivity: {e}")
            driver.close()
            raise
        
        return driver
    
    @classmethod
    def close(cls) -> None:
        """
        Close the shared driver and release all connections.
        
        Should be called during application shutdown.
        Thread-safe - can be called from any thread.
        """
        with cls._lock:
            if cls._driver is not None:
                logger.info("Closing shared Neo4j driver")
                try:
                    cls._driver.close()
                except Exception as e:
                    logger.warning(f"Error closing Neo4j driver: {e}")
                finally:
                    cls._driver = None
    
    @classmethod
    def is_connected(cls) -> bool:
        """
        Check if the driver is connected and healthy.
        
        Returns:
            True if driver exists and can connect to Neo4j
        """
        if cls._driver is None:
            return False
        
        try:
            cls._driver.verify_connectivity()
            return True
        except Exception:
            return False
    
    @classmethod
    def get_pool_status(cls) -> dict:
        """
        Get current connection pool status for monitoring.
        
        Useful for debugging connection issues and monitoring.
        
        Returns:
            Dict with pool metrics (varies by Neo4j driver version)
        """
        if cls._driver is None:
            return {"initialized": False}
        
        return {
            "initialized": True,
            "max_pool_size": cls.MAX_CONNECTION_POOL_SIZE,
            "max_connection_lifetime": cls.MAX_CONNECTION_LIFETIME,
            "connected": cls.is_connected(),
        }


def get_neo4j_driver() -> Driver:
    """
    Convenience function to get the shared Neo4j driver.
    
    This is the primary interface for tools to obtain a driver.
    
    Example:
        from .neo4j_driver_manager import get_neo4j_driver
        
        driver = get_neo4j_driver()
        with driver.session() as session:
            result = session.run("MATCH (n) RETURN n LIMIT 1")
    """
    return Neo4jDriverManager.get_driver()


def close_neo4j_driver() -> None:
    """
    Close the shared Neo4j driver.
    
    Should be called during application shutdown.
    """
    Neo4jDriverManager.close()

atexit.register(close_neo4j_driver)
