import pytest
import httpx
from bson import ObjectId

@pytest.mark.asyncio
async def test_document_lifecycle():
    base_url = "http://localhost:8000"
    collection_name = "testCollection"
    headers = {"Authorization": f"Bearer token"}
    
    # Use httpx.AsyncClient to handle async requests
    async with httpx.AsyncClient() as client:
        # Create a document
        document = {"name": "Test Document", "value": "Test Value"}
        create_response = await client.post(f"{base_url}/documents/{collection_name}", json=document, headers=headers)
        assert create_response.status_code == 200
        document_id = create_response.json()["id"]
        assert ObjectId.is_valid(document_id)

        # Read the document
        read_response = await client.get(f"{base_url}/documents/{collection_name}/{document_id}", headers=headers)
        assert read_response.status_code == 200
        read_data = read_response.json()
        assert read_data["_id"] == document_id
        assert read_data["name"] == document["name"]
        assert read_data["value"] == document["value"]

        # Update the document
        update_fields = {"value": "Updated Test Value"}
        update_response = await client.put(f"{base_url}/documents/{collection_name}/{document_id}", json=update_fields, headers=headers)
        assert update_response.status_code == 200
        updated_data = update_response.json()
        assert updated_data["value"] == update_fields["value"]

        # Delete the document
        delete_response = await client.delete(f"{base_url}/documents/{collection_name}/{document_id}", headers=headers)
        assert delete_response.status_code == 200
        delete_data = delete_response.json()
        assert delete_data["message"] == "Document deleted successfully"

        # Verify deletion
        verify_delete_response = await client.get(f"{base_url}/documents/{collection_name}/{document_id}", headers=headers)
        assert verify_delete_response.status_code == 404
