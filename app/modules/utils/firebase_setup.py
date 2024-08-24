import base64
import json
import logging
import os

import firebase_admin
from firebase_admin import credentials


class FirebaseSetup:
    @staticmethod
    def firebase_init():
        service_account_base64 = os.getenv("FIREBASE_SERVICE_ACCOUNT")
        current_file_path = os.path.dirname(os.path.abspath(__file__))
        parent_directory = os.path.abspath(
            os.path.join(current_file_path, *([".."] * 3))
        )
        cred = None
        if service_account_base64:
            try:
                service_account_info = base64.b64decode(service_account_base64).decode(
                    "utf-8"
                )
                service_account_json = json.loads(service_account_info)
                cred = credentials.Certificate(service_account_json)
                logging.info("Loaded Firebase credentials from environment variable.")
            except Exception as e:
                logging.info(
                    f"Error decoding Firebase service account from environment variable: {e}"
                )
                cred = credentials.Certificate(
                    os.path.join(parent_directory, "firebase_service_account.json")
                )
                logging.info("Loaded Firebase credentials from local file as fallback.")
        else:
            cred = credentials.Certificate(
                os.path.join(parent_directory, "firebase_service_account.json")
            )
            logging.info("Loaded Firebase credentials from local file.")

        firebase_admin.initialize_app(cred)
