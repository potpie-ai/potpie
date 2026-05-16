import os
import pytest
from cryptography.fernet import Fernet
from fastapi import HTTPException
from unittest.mock import patch

from app.modules.key_management.secret_manager import SecretStorageHandler

# Three stable Fernet keys used across tests.
KEY_V1 = "OiEqE7Mstu46OFOh-oV7llA9Sz1Vm9AQog_An3COdY8="
KEY_V2 = "FaXISSDJ3bncymQPNw8x1Z57X_IV6-zn6S8rChL7KGE="
KEY_V3 = "Zy2xW9cKJqBn8mDrFtPuLsVeHiAo3G4N7Y0X5CkQjE6="


class TestSecretStorageHandlerEncryption:
    """Tests for encrypt_value / decrypt_value and key rotation behaviour."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _encrypt_raw(key: str, plaintext: str) -> str:
        """Encrypt directly with a given key, without any version prefix.
        Simulates blobs that were written before key rotation was introduced."""
        return Fernet(key.encode()).encrypt(plaintext.encode()).decode()

    # ------------------------------------------------------------------
    # Basic round-trip
    # ------------------------------------------------------------------

    def test_encrypt_produces_version_prefix(self):
        with patch.dict(os.environ, {"SECRET_ENCRYPTION_KEY": KEY_V1}):
            result = SecretStorageHandler.encrypt_value("hello")
        assert result.startswith("v1:"), f"Expected 'v1:' prefix, got: {result[:10]}"

    def test_round_trip_single_key(self):
        plaintext = "my-secret-api-key"
        with patch.dict(os.environ, {"SECRET_ENCRYPTION_KEY": KEY_V1}):
            encrypted = SecretStorageHandler.encrypt_value(plaintext)
            decrypted = SecretStorageHandler.decrypt_value(encrypted)
        assert decrypted == plaintext

    # ------------------------------------------------------------------
    # Key rotation — versioned blobs
    # ------------------------------------------------------------------

    def test_rotation_new_writes_use_new_version_label(self):
        """After rotation, new writes carry the new active version label."""
        plaintext = "new-secret"
        with patch.dict(
            os.environ,
            {
                "SECRET_ENCRYPTION_KEY": KEY_V2,
                "SECRET_ENCRYPTION_KEY_ACTIVE_VERSION": "v2",
                "SECRET_ENCRYPTION_KEY_v1": KEY_V1,
            },
        ):
            encrypted = SecretStorageHandler.encrypt_value(plaintext)
            assert encrypted.startswith("v2:")
            decrypted = SecretStorageHandler.decrypt_value(encrypted)
        assert decrypted == plaintext

    def test_rotation_old_versioned_blob_still_decrypts(self):
        """A v1 blob encrypted with the old key must still decrypt after rotation."""
        plaintext = "old-secret"
        # Simulate a blob written when KEY_V1 was the active key at label "v1".
        with patch.dict(os.environ, {"SECRET_ENCRYPTION_KEY": KEY_V1}):
            old_blob = SecretStorageHandler.encrypt_value(plaintext)
        assert old_blob.startswith("v1:")

        # Rotate: KEY_V2 becomes active at "v2"; KEY_V1 is preserved at "v1".
        with patch.dict(
            os.environ,
            {
                "SECRET_ENCRYPTION_KEY": KEY_V2,
                "SECRET_ENCRYPTION_KEY_ACTIVE_VERSION": "v2",
                "SECRET_ENCRYPTION_KEY_v1": KEY_V1,
            },
        ):
            decrypted = SecretStorageHandler.decrypt_value(old_blob)
        assert decrypted == plaintext

    def test_multi_generation_rotation(self):
        """Three-generation rotation: v1, v2, and v3 blobs all decrypt under the v3-active ring."""
        # Create blobs at each prior generation.
        with patch.dict(os.environ, {"SECRET_ENCRYPTION_KEY": KEY_V1}):
            blob_v1 = SecretStorageHandler.encrypt_value("secret-v1")
        assert blob_v1.startswith("v1:")

        with patch.dict(
            os.environ,
            {
                "SECRET_ENCRYPTION_KEY": KEY_V2,
                "SECRET_ENCRYPTION_KEY_ACTIVE_VERSION": "v2",
                "SECRET_ENCRYPTION_KEY_v1": KEY_V1,
            },
        ):
            blob_v2 = SecretStorageHandler.encrypt_value("secret-v2")
        assert blob_v2.startswith("v2:")

        # Rotate to v3: all three generations must decrypt, and new writes use v3.
        with patch.dict(
            os.environ,
            {
                "SECRET_ENCRYPTION_KEY": KEY_V3,
                "SECRET_ENCRYPTION_KEY_ACTIVE_VERSION": "v3",
                "SECRET_ENCRYPTION_KEY_v1": KEY_V1,
                "SECRET_ENCRYPTION_KEY_v2": KEY_V2,
            },
        ):
            assert SecretStorageHandler.decrypt_value(blob_v1) == "secret-v1"
            assert SecretStorageHandler.decrypt_value(blob_v2) == "secret-v2"
            blob_v3 = SecretStorageHandler.encrypt_value("secret-v3")
            assert blob_v3.startswith("v3:")
            assert SecretStorageHandler.decrypt_value(blob_v3) == "secret-v3"

    # ------------------------------------------------------------------
    # Backward compatibility — legacy blobs (no version prefix)
    # ------------------------------------------------------------------

    def test_legacy_blob_decrypts_with_single_key(self):
        """Unversioned blobs written before this feature was introduced must still work."""
        plaintext = "legacy-secret"
        legacy_blob = self._encrypt_raw(KEY_V1, plaintext)

        with patch.dict(os.environ, {"SECRET_ENCRYPTION_KEY": KEY_V1}):
            decrypted = SecretStorageHandler.decrypt_value(legacy_blob)
        assert decrypted == plaintext

    def test_legacy_blob_decrypts_after_rotation(self):
        """After rotating, legacy blobs encrypted with the old key must still decrypt."""
        plaintext = "legacy-after-rotation"
        legacy_blob = self._encrypt_raw(KEY_V1, plaintext)

        # Rotate: KEY_V2 active; KEY_V1 preserved at v1 for legacy blobs.
        with patch.dict(
            os.environ,
            {
                "SECRET_ENCRYPTION_KEY": KEY_V2,
                "SECRET_ENCRYPTION_KEY_ACTIVE_VERSION": "v2",
                "SECRET_ENCRYPTION_KEY_v1": KEY_V1,
            },
        ):
            decrypted = SecretStorageHandler.decrypt_value(legacy_blob)
        assert decrypted == plaintext

    # ------------------------------------------------------------------
    # Error cases
    # ------------------------------------------------------------------

    def test_missing_key_raises_500(self):
        env = {k: v for k, v in os.environ.items() if k != "SECRET_ENCRYPTION_KEY"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(HTTPException) as exc_info:
                SecretStorageHandler.encrypt_value("anything")
        assert exc_info.value.status_code == 500

    def test_wrong_key_raises_500(self):
        """A versioned blob that cannot be decrypted by any ring key raises 500."""
        plaintext = "secret"
        with patch.dict(os.environ, {"SECRET_ENCRYPTION_KEY": KEY_V1}):
            encrypted = SecretStorageHandler.encrypt_value(plaintext)  # v1:<token_KEY_V1>

        # Try to decrypt with KEY_V2 only — KEY_V1 is not in the ring at all.
        with patch.dict(os.environ, {"SECRET_ENCRYPTION_KEY": KEY_V2}):
            with pytest.raises(HTTPException) as exc_info:
                SecretStorageHandler.decrypt_value(encrypted)
        assert exc_info.value.status_code == 500

    def test_invalid_key_format_raises_500(self):
        with patch.dict(os.environ, {"SECRET_ENCRYPTION_KEY": "not-a-valid-fernet-key"}):
            with pytest.raises(HTTPException) as exc_info:
                SecretStorageHandler.encrypt_value("anything")
        assert exc_info.value.status_code == 500

    def test_legacy_blob_wrong_key_raises_500(self):
        """A legacy (unversioned) blob that cannot be decrypted by any ring key raises 500."""
        legacy_blob = self._encrypt_raw(KEY_V1, "secret")

        # Ring contains only KEY_V2 — KEY_V1 is nowhere in the ring.
        with patch.dict(os.environ, {"SECRET_ENCRYPTION_KEY": KEY_V2}):
            with pytest.raises(HTTPException) as exc_info:
                SecretStorageHandler.decrypt_value(legacy_blob)
        assert exc_info.value.status_code == 500

    def test_invalid_old_key_in_env_does_not_crash(self):
        """A malformed SECRET_ENCRYPTION_KEY_v* value is silently skipped; active key still works."""
        plaintext = "still-works"
        with patch.dict(
            os.environ,
            {
                "SECRET_ENCRYPTION_KEY": KEY_V1,
                "SECRET_ENCRYPTION_KEY_v2": "not-a-valid-fernet-key",
            },
        ):
            encrypted = SecretStorageHandler.encrypt_value(plaintext)
            assert encrypted.startswith("v1:")
            assert SecretStorageHandler.decrypt_value(encrypted) == plaintext
