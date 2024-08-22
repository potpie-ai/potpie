import os
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from typing import Optional

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

    def _connect(self):
        try:
            self._client = MongoClient(
                os.environ.get("MONGODB_URI"),
                maxPoolSize=50,  # Adjust based on your needs
                waitQueueTimeoutMS=2500  # Adjust based on your needs
            )
            self._db = self._client[os.environ.get("MONGODB_DB_NAME")]
            # Ping the server to check the connection
            self._client.admin.command('ping')
        except ConnectionFailure:
            print("Server not available")
            raise

    def get_collection(self, collection_name: str):
        return self._db[collection_name]

    def put(self, collection_name: str, document_id: str, data: dict):
        collection = self.get_collection(collection_name)
        collection.update_one(
            {"_id": document_id},
            {"$set": data},
            upsert=True
        )

    def get(self, collection_name: str, document_id: str):
        collection = self.get_collection(collection_name)
        return collection.find_one({"_id": document_id})

    def delete(self, collection_name: str, document_id: str):
        collection = self.get_collection(collection_name)
        collection.delete_one({"_id": document_id})

    def close(self):
        if self._client:
            self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()