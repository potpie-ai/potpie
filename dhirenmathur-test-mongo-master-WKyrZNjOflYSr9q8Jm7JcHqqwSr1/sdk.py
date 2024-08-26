from typing import Any, Dict
import httpx
from pydantic import BaseModel

class FastAPIMongoSDK:
    def __init__(self, base_url: str, bearer_token: str):
        self.base_url = base_url
        self.bearer_token = bearer_token
        self.client = httpx.AsyncClient()

    async def create_document(self, collection_name: str, document: Dict[str, Any]):
        response = await self.client.post(
            f"{self.base_url}/documents/{collection_name}",
            json=document,
            headers={"Authorization": f"Bearer {self.bearer_token}"}
        )
        return response.json()

    async def read_document(self, collection_name: str, document_id: str):
        response = await self.client.get(
            f"{self.base_url}/documents/{collection_name}/{document_id}",
            headers={"Authorization": f"Bearer {self.bearer_token}"}
        )
        return response.json()

    async def update_document(self, collection_name: str, document_id: str, update_fields: Dict[str, Any]):
        response = await self.client.put(
            f"{self.base_url}/documents/{collection_name}/{document_id}",
            json=update_fields,
            headers={"Authorization": f"Bearer {self.bearer_token}"}
        )
        return response.json()

    async def delete_document(self, collection_name: str, document_id: str):
        response = await self.client.delete(
            f"{self.base_url}/documents/{collection_name}/{document_id}",
            headers={"Authorization": f"Bearer {self.bearer_token}"}
        )
        return response.json()
