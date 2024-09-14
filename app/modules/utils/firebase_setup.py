import os
import firebase_admin
from firebase_admin import auth, credentials
import base64
import json
import logging

def firebase_init():
    # Construct the path to the text file containing the Base64-encoded credentials.
    base64_file_path = os.path.join(os.getcwd(), 'firebase_service_account.txt')

    # Check if the Base64 encoded file exists.
    if os.path.exists(base64_file_path):
        try:
            with open(base64_file_path, 'r') as file:
                service_account_base64 = file.read()
            
            # Decode the Base64 content and parse it as JSON.
            service_account_info = base64.b64decode(service_account_base64).decode('utf-8')
            service_account_json = json.loads(service_account_info)

            # Use the decoded JSON to initialize Firebase credentials.
            cred = credentials.Certificate(service_account_json)
            logging.info("Loaded Firebase credentials from Base64 encoded file.")
        except Exception as e:
            logging.error(f"Error decoding or parsing Firebase service account from Base64 file: {e}")
            raise
    else:
        logging.error("Firebase service account file 'firebase_service_account.txt' not found.")
        raise FileNotFoundError("Firebase service account file 'firebase_service_account.txt' not found.")

    # Initialize the Firebase app with the credentials.
    firebase_admin.initialize_app(cred)
