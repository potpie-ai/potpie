"""
Token encryption utilities for secure storage of OAuth tokens
"""

import os
import base64
from cryptography.fernet import Fernet
import logging


class TokenEncryption:
    """Handles encryption and decryption of OAuth tokens for secure storage"""

    def __init__(self):
        self._fernet = None
        self._initialize_fernet()

    def _initialize_fernet(self):
        """Initialize Fernet encryption with key from environment"""
        try:
            # Get encryption key from environment
            encryption_key = os.getenv("ENCRYPTION_KEY")

            if not encryption_key:
                # Generate a new key if none exists (for development)
                logging.warning(
                    "ENCRYPTION_KEY not found in environment. Generating new key for development."
                )
                encryption_key = Fernet.generate_key().decode()
                logging.warning(f"Generated encryption key: {encryption_key}")
                logging.warning(
                    "IMPORTANT: Set ENCRYPTION_KEY environment variable in production!"
                )

            # Ensure key is properly formatted
            if isinstance(encryption_key, str):
                encryption_key = encryption_key.encode()

            # Pad or truncate key to 32 bytes if needed
            if len(encryption_key) != 32:
                if len(encryption_key) < 32:
                    encryption_key = encryption_key.ljust(32, b"0")
                else:
                    encryption_key = encryption_key[:32]

            # Create Fernet instance
            self._fernet = Fernet(base64.urlsafe_b64encode(encryption_key))

        except Exception as e:
            logging.error(f"Failed to initialize token encryption: {str(e)}")
            raise Exception(f"Token encryption initialization failed: {str(e)}")

    def encrypt_token(self, token: str) -> str:
        """Encrypt a token for secure storage"""
        try:
            if not token:
                return ""

            if not self._fernet:
                raise Exception("Token encryption not initialized")

            encrypted_bytes = self._fernet.encrypt(token.encode())
            return encrypted_bytes.decode()

        except Exception as e:
            logging.error(f"Failed to encrypt token: {str(e)}")
            raise Exception(f"Token encryption failed: {str(e)}")

    def decrypt_token(self, encrypted_token: str) -> str:
        """Decrypt a token for use"""
        try:
            if not encrypted_token:
                return ""

            if not self._fernet:
                raise Exception("Token encryption not initialized")

            decrypted_bytes = self._fernet.decrypt(encrypted_token.encode())
            return decrypted_bytes.decode()

        except Exception as e:
            logging.error(f"Failed to decrypt token: {str(e)}")
            raise Exception(f"Token decryption failed: {str(e)}")


# Global instance
token_encryption = TokenEncryption()


def encrypt_token(token: str) -> str:
    """Encrypt token for secure storage"""
    return token_encryption.encrypt_token(token)


def decrypt_token(encrypted_token: str) -> str:
    """Decrypt token for use"""
    return token_encryption.decrypt_token(encrypted_token)
