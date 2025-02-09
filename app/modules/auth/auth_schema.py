from pydantic import BaseModel


class LoginRequest(BaseModel):
    email: str
    password: str


class User(BaseModel):
    email: str
    user_id: str
