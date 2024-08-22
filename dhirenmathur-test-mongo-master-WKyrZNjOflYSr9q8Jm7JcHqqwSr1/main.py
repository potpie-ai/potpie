from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.encoders import jsonable_encoder
from fastapi.security.http import HTTPBearer, HTTPAuthorizationCredentials
from firebase import verify_id_token
from mongo import MongoDB
from typing import Dict

app = FastAPI()
db = MongoDB()
auth_scheme = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Security(auth_scheme)):
    token = credentials.credentials
    decoded_token = verify_id_token(token)
    if not decoded_token:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    return decoded_token


@app.post("/documents/{collection_name}", dependencies=[Depends(get_current_user)])
async def create_document(collection_name: str, document: Dict):
    document_id = await db.create_document(collection_name, document)
    return {"id": document_id}

@app.get("/documents/{collection_name}/{document_id}", dependencies=[Depends(get_current_user)])
async def read_document(collection_name: str, document_id: str):
    document = await db.read_document(collection_name, document_id)
    print("Test")
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    
    document['_id'] = str(document['_id'])  # Convert ObjectId to string
    return jsonable_encoder(document)

@app.put("/documents/{collection_name}/{document_id}", dependencies=[Depends(get_current_user)])
async def update_document(collection_name: str, document_id: str, update_fields: Dict):
    updated_document = await db.update_document(collection_name, document_id, update_fields)
    return updated_document

@app.delete("/documents/{collection_name}/{document_id}", dependencies=[Depends(get_current_user)])
async def delete_document(collection_name: str, document_id: str):
    await db.delete_document(collection_name, document_id)
    return {"message": "Document deleted successfully"}
