"""
Token encryption utilities for secure storage of OAuth tokens
"""

import os
import base64
import hashlib
from cryptography.fernet import Fernet
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


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
                # Auto-generate and persist key to file (works in both dev and prod)
                key_file = ".encryption_key_persistent"
                if os.path.exists(key_file):
                    # Load existing persisted key
                    with open(key_file, 'r') as f:
                        encryption_key = f.read().strip()
                    logger.info(f"Loaded persisted encryption key from {key_file}")
                else:
                    # Generate new key and persist it
                    encryption_key = Fernet.generate_key().decode()
                    # Create file with restrictive permissions (owner read/write only)
                    fd = os.open(key_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                    with os.fdopen(fd, 'w') as f:
                        f.write(encryption_key)
                    logger.warning(
                        f"Generated and saved encryption key to {key_file}"
                    )
                    # Log a non-reversible fingerprint for debugging
                    key_fingerprint = hashlib.sha256(encryption_key.encode()).hexdigest()[:8]
                    logger.debug(f"Key fingerprint: {key_fingerprint}")

                logger.warning(
                    "IMPORTANT: For better security, set ENCRYPTION_KEY environment variable!"
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
            logger.exception("Failed to initialize token encryption")
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
            logger.exception("Failed to encrypt token")
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
            # Try OLD_ENCRYPTION_KEY for key rotation support
            old_key = os.getenv("OLD_ENCRYPTION_KEY")
            if old_key:
                try:
                    logger.info("Attempting decryption with OLD_ENCRYPTION_KEY")
                    # Format old key
                    if isinstance(old_key, str):
                        old_key = old_key.encode()
                    if len(old_key) != 32:
                        if len(old_key) < 32:
                            old_key = old_key.ljust(32, b"0")
                        else:
                            old_key = old_key[:32]

                    old_fernet = Fernet(base64.urlsafe_b64encode(old_key))
                    decrypted_bytes = old_fernet.decrypt(encrypted_token.encode())
                    logger.warning(
                        "Token was encrypted with old key. "
                        "It will be re-encrypted with new key on next update."
                    )
                    return decrypted_bytes.decode()
                except Exception as old_key_error:
                    logger.debug(f"OLD_ENCRYPTION_KEY also failed: {old_key_error}")

            # Both current and old keys failed
            logger.exception("Failed to decrypt token with current and old keys")
            raise Exception(f"Token decryption failed: {str(e)}")


# Global instance
token_encryption = TokenEncryption()


def encrypt_token(token: str) -> str:
    """Encrypt token for secure storage"""
    return token_encryption.encrypt_token(token)


def decrypt_token(encrypted_token: str) -> str:
    """Decrypt token for use"""
    return token_encryption.decrypt_token(encrypted_token)
