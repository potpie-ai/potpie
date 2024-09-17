import base64
import json
import logging
import os

import firebase_admin
from firebase_admin import credentials


class FirebaseSetup:
    @staticmethod
    def firebase_init():
        # Construct the paths for both file types.
        base64_file_path = os.path.join(os.getcwd(), "firebase_service_account.txt")
        json_file_path = os.path.join(os.getcwd(), "firebase_service_account.json")

        try:
            # Check if the Base64 encoded file exists and read it.
            if os.path.exists(base64_file_path):
                with open(base64_file_path, "r") as file:
                    service_account_base64 = file.read()

                # Decode the Base64 content and parse it as JSON.
                service_account_info = base64.b64decode(service_account_base64).decode(
                    "utf-8"
                )
                service_account_json = json.loads(service_account_info)
                logging.info("Loaded Firebase credentials from Base64 encoded file.")
            elif os.path.exists(json_file_path):
                # If the Base64 file does not exist, check for the JSON file.
                with open(json_file_path, "r") as file:
                    service_account_json = json.load(file)
                logging.info("Loaded Firebase credentials from JSON file.")
            else:
                logging.error(
                    "Neither Firebase service account file 'firebase_service_account.txt' nor 'firebase_service_account.json' found."
                )
                raise FileNotFoundError(
                    "Neither Firebase service account file 'firebase_service_account.txt' nor 'firebase_service_account.json' found."
                )

            # Use the decoded or directly loaded JSON to initialize Firebase credentials.
            cred = credentials.Certificate(service_account_json)

            # Initialize the Firebase app with the credentials.
            firebase_admin.initialize_app(cred)

        except Exception as e:
            logging.error(f"Error loading Firebase service account credentials: {e}")
            raise
