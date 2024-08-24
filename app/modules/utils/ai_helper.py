import os

from langchain_openai.chat_models import ChatOpenAI
from portkey_ai import PORTKEY_GATEWAY_URL, createHeaders

from app.core.mongo_manager import MongoDBHelper
from app.modules.key_management.secret_manager import get_secret


class AIHelper:
    @staticmethod
    def get_llm_client(user_id, model_name):
        provider_key = AIHelper.get_provider_key(user_id)
        return AIHelper.create_client(
            provider_key["provider"], provider_key["key"], model_name, user_id
        )

    @staticmethod
    def get_provider_key(customer_id):
        if os.environ.get("isDevelopmentMode") == "enabled":
            return {"provider": "openai", "key": os.environ.get("OPENAI_API_KEY")}
        mongo_helper = MongoDBHelper()
        preference = mongo_helper.get(customer_id, "preferences").get()
        if preference.exists and preference.get("provider") == "openai":
            return {
                "provider": "openai",
                "key": get_secret("openai", customer_id)["api_key"],
            }
        else:
            return {"provider": "openai", "key": os.environ.get("OPENAI_API_KEY")}

    @staticmethod
    def create_client(provider, key, model_name, user_id):
        if provider == "openai":
            PROVIDER_API_KEY = key

            if os.getenv("isDevelopmentMode") == "enabled":
                return ChatOpenAI(api_key=PROVIDER_API_KEY, model=model_name)
            else:
                PORTKEY_API_KEY = os.environ.get("PORTKEY_API_KEY")
                portkey_headers = createHeaders(
                    api_key=PORTKEY_API_KEY,
                    provider="openai",
                    metadata={"_user": user_id, "environment": os.environ.get("ENV")},
                )
                return ChatOpenAI(
                    api_key=PROVIDER_API_KEY,
                    model=model_name,
                    base_url=PORTKEY_GATEWAY_URL,
                    default_headers=portkey_headers,
                )
