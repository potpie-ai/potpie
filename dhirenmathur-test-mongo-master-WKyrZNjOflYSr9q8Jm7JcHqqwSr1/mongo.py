from motor.motor_asyncio import AsyncIOMotorClient
from typing import List, Dict
from bson import ObjectId
import os


class MongoDB:
    def __init__(self):
        self.client = AsyncIOMotorClient(os.environ["MONGODB_URI"])
        self.db = self.client[os.environ["MONGODB_DB_NAME"]]

    def document_to_json(self, document):
        if document:
            # Convert ObjectId to str and prepare for JSON serialization
            document["_id"] = str(document["_id"])
        return document

    async def create_document(self, collection_name: str, document: Dict):
        collection = self.db[collection_name]
        result = await collection.insert_one(document)
        return str(result.inserted_id)

    async def read_document(self, collection_name: str, document_id: str):
        collection = self.db[collection_name]
        document = await collection.find_one({"_id": ObjectId(document_id)})
        return self.document_to_json(document)  

    async def update_document(
        self, collection_name: str, document_id: str, update_fields: Dict
    ):
        collection = self.db[collection_name]
        await collection.update_one(
            {"_id": ObjectId(document_id)}, {"$set": update_fields}
        )
        updated_document = await collection.find_one({"_id": ObjectId(document_id)})
        return self.document_to_json(updated_document)

    async def delete_document(self, collection_name: str, document_id: str):
        collection = self.db[collection_name]
        await collection.delete_one({"_id": ObjectId(document_id)})
