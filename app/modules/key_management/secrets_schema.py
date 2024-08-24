from pydantic import BaseModel, validator
from typing import Literal
import re    

def validate_openai_api_key_format(api_key: str) -> bool:
    pattern = r"^sk-[a-zA-Z0-9]{48}$"
    proj_pattern = r"^sk-proj-[a-zA-Z0-9]{48}$"
    return bool(re.match(pattern, api_key)) or bool(re.match(proj_pattern, api_key))

class BaseSecretRequest(BaseModel):
    api_key: str
    provider: Literal["openai"] = "openai"

    @validator("api_key")
    def api_key_format(cls, v: str) -> str:
        if not validate_openai_api_key_format(v):
            raise ValueError("Invalid OpenAI API key format")
        return v

class UpdateSecretRequest(BaseSecretRequest):
    pass

class CreateSecretRequest(BaseSecretRequest):
    pass