import logging
import os
from typing import Optional

import certifi
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure


class MongoManager:
    _instance = None
    _client: Optional[MongoClient] = None
    _db = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        if self._instance is not None:
            raise RuntimeError("Use get_instance() to get the MongoManager instance")
        self._connect()

    def get_mongo_connection(self):
        try:
            db_name = os.getenv("MONGODB_DB_NAME", "test")
            mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
            env = os.getenv(
                "ENV", "development"
            )  # Assume development if ENV is not set

            if not mongo_uri:
                raise ValueError("MONGO_URI environment variable is not set")

            if not db_name:
                raise ValueError("MONGODB_DB_NAME environment variable is not set")

            # Establish the connection based on the environment
            if env in ["production", "staging"]:
                self.client = MongoClient(
                    mongo_uri,
                    maxPoolSize=50,
                    waitQueueTimeoutMS=2500,
                    tlsCAFile=certifi.where(),
                )
            else:
                self.client = MongoClient(
                    mongo_uri,
                    maxPoolSize=50,
                    waitQueueTimeoutMS=2500,
                )

            # Return the established connection
            db_connection = self.client[db_name]
            db_connection.command("ping")  # Verify the connection
            return self.client

        except (ConnectionFailure, ValueError) as e:
            logging.error(f"Failed to connect to MongoDB: {str(e)}")
            raise

    def _connect(self):
        if self._client is None:
            try:
                # Establish connection using the utility function
                self._client = self.get_mongo_connection()

                db_name = os.environ.get("MONGODB_DB_NAME")
                if not db_name:
                    raise ValueError("MONGODB_DB_NAME environment variable is not set")

                self._db = self._client[db_name]

                # Verify the connection and database
                self.verify_connection()

            except (ConnectionFailure, ValueError) as e:
                logging.error(f"Failed to connect to MongoDB: {str(e)}")
                raise

    def verify_connection(self):
        try:
            # Ping the server to check the connection
            self._client.admin.command("ping")

            # List all collections to verify database access
            self._db.list_collection_names()

            logging.info(
                "Successfully connected to MongoDB and verified database access"
            )
        except OperationFailure as e:
            logging.error(f"Failed to verify MongoDB connection: {str(e)}")
            raise

    def get_collection(self, collection_name: str):
        self._connect()  # Ensure connection is established
        return self._db[collection_name]

    def put(self, collection_name: str, document_id: str, data: dict):
        try:
            collection = self.get_collection(collection_name)
            result = collection.update_one(
                {"_id": document_id}, {"$set": data}, upsert=True
            )
            logging.info(
                f"Document {'updated' if result.modified_count else 'inserted'} in {collection_name}"
            )
            return result
        except Exception as e:
            logging.error(f"Failed to put document in {collection_name}: {str(e)}")
            raise

    def get(self, collection_name: str, document_id: str):
        try:
            collection = self.get_collection(collection_name)
            document = collection.find_one({"_id": document_id})
            if document:
                logging.info(f"Document retrieved from {collection_name}")
            else:
                logging.info(f"Document not found in {collection_name}")
            return document
        except Exception as e:
            logging.error(f"Failed to get document from {collection_name}: {str(e)}")
            raise

    def delete(self, collection_name: str, document_id: str):
        try:
            collection = self.get_collection(collection_name)
            result = collection.delete_one({"_id": document_id})
            if result.deleted_count:
                logging.info(f"Document deleted from {collection_name}")
            else:
                logging.info(f"Document not found in {collection_name}")
            return result
        except Exception as e:
            logging.error(f"Failed to delete document from {collection_name}: {str(e)}")
            raise

    def close(self):
        if self._client:
            self._client.close()
            self._client = None
            self._db = None
            logging.info("MongoDB connection closed")

    def reconnect(self):
        self.close()
        self._connect()
        logging.info("Reconnected to MongoDB")

    @classmethod
    def close_connection(cls):
        if cls._instance:
            cls._instance.close()
            cls._instance = None
            logging.info("MongoDB connection closed and instance reset")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass  # Don't close the connection here
