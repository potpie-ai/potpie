import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


class Utility:
    @staticmethod
    def base_url() -> str:
        return "http://localhost:8001"

    @staticmethod
    def get_user_id() -> str:
        return os.getenv("defaultUsername", "defaultuser")
