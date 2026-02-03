"""
Unified storage client abstraction for multi-cloud object storage.

StorageClient defines the interface; S3StorageClient and AzureBlobStorageClient
provide the implementations.
"""

import base64
import datetime
import hashlib
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Literal, TypedDict

import boto3
from botocore.exceptions import ClientError
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Descriptor TypedDicts — used by storage strategies, consumed here
# ---------------------------------------------------------------------------


class S3Descriptor(TypedDict):
    provider: str
    client_type: Literal["s3"]
    bucket_name: str
    client_kwargs: Dict[str, Any]


class AzureDescriptor(TypedDict):
    provider: str
    client_type: Literal["azure"]
    bucket_name: str
    azure_account_name: str
    azure_account_key: str


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class StorageClientError(Exception):
    """Raised when a storage operation fails."""

    pass


class BlobNotFoundError(StorageClientError):
    """Raised when a requested blob/object does not exist."""

    pass


# ---------------------------------------------------------------------------
# Retry helpers
# ---------------------------------------------------------------------------


def _is_transient(exc: BaseException) -> bool:
    """Return True for errors that are likely transient and worth retrying."""
    # boto3 transient errors
    if isinstance(exc, ClientError):
        code = exc.response.get("Error", {}).get("Code", "")
        return code in (
            "RequestTimeout",
            "ThrottlingException",
            "SlowDown",
            "ServiceUnavailable",
            "InternalError",
        )
    # Azure transient errors
    try:
        from azure.core.exceptions import (
            HttpResponseError,
            ServiceRequestError,
            ServiceResponseError,
        )

        if isinstance(exc, (ServiceRequestError, ServiceResponseError)):
            return True
        if isinstance(exc, HttpResponseError) and exc.status_code in (
            429,
            500,
            502,
            503,
            504,
        ):
            return True
    except ImportError:
        pass
    # Generic network errors
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    return False


_retry_transient = retry(
    retry=retry_if_exception(_is_transient),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------


class StorageClient(ABC):
    """Abstract interface for object storage operations.

    All operations are synchronous.  ``delete`` is idempotent — deleting a
    non-existent object is a no-op (no exception raised).
    """

    @abstractmethod
    def check_bucket(self, bucket_name: str, create_if_missing: bool = True) -> None:
        """Verify the bucket/container exists and is accessible.
        
        If create_if_missing is True (default), creates the bucket/container
        if it doesn't exist. Raises StorageClientError on failure."""

    @abstractmethod
    def upload(
        self,
        bucket_name: str,
        key: str,
        data: bytes,
        content_type: str,
    ) -> None:
        """Upload bytes to the given key. Raises StorageClientError on failure."""

    @abstractmethod
    def download(self, bucket_name: str, key: str) -> bytes:
        """Download and return bytes. Raises BlobNotFoundError if missing."""

    @abstractmethod
    def delete(self, bucket_name: str, key: str) -> None:
        """Delete the object. Idempotent — does not raise if the object is
        already gone."""

    @abstractmethod
    def generate_signed_url(
        self, bucket_name: str, key: str, expires_in_seconds: int
    ) -> str:
        """Return a time-limited download URL."""


# ---------------------------------------------------------------------------
# S3 implementation (also covers GCS via S3-interop)
# ---------------------------------------------------------------------------


class S3StorageClient(StorageClient):
    """StorageClient backed by boto3 (AWS S3 and S3-compatible endpoints like GCS)."""

    def __init__(self, client_kwargs: Dict[str, Any]):
        self._client = boto3.client("s3", **client_kwargs)

    def check_bucket(self, bucket_name: str, create_if_missing: bool = True) -> None:
        try:
            self._client.head_bucket(Bucket=bucket_name)
            logger.debug("Bucket check passed: %s", bucket_name)
        except Exception as e:
            raise StorageClientError(f"Bucket check failed: {e}") from e

    @_retry_transient
    def upload(
        self,
        bucket_name: str,
        key: str,
        data: bytes,
        content_type: str,
    ) -> None:
        try:
            md5_b64 = base64.b64encode(hashlib.md5(data).digest()).decode("utf-8")
            self._client.put_object(
                Bucket=bucket_name,
                Key=key,
                Body=data,
                ContentType=content_type,
                ContentMD5=md5_b64,
            )
            logger.debug("Uploaded %s/%s (%d bytes)", bucket_name, key, len(data))
        except Exception as e:
            raise StorageClientError(f"Upload failed: {e}") from e

    @_retry_transient
    def download(self, bucket_name: str, key: str) -> bytes:
        try:
            response = self._client.get_object(Bucket=bucket_name, Key=key)
            data = response["Body"].read()
            logger.debug("Downloaded %s/%s (%d bytes)", bucket_name, key, len(data))
            return data
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") == "NoSuchKey":
                raise BlobNotFoundError(f"Object not found: {key}") from e
            raise StorageClientError(f"Download failed: {e}") from e

    @_retry_transient
    def delete(self, bucket_name: str, key: str) -> None:
        """S3 delete_object is already idempotent — returns 204 even for
        non-existent keys."""
        try:
            self._client.delete_object(Bucket=bucket_name, Key=key)
            logger.debug("Deleted %s/%s", bucket_name, key)
        except Exception as e:
            raise StorageClientError(f"Delete failed: {e}") from e

    @_retry_transient
    def generate_signed_url(
        self, bucket_name: str, key: str, expires_in_seconds: int
    ) -> str:
        try:
            url = self._client.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket_name, "Key": key},
                ExpiresIn=expires_in_seconds,
            )
            logger.debug(
                "Generated signed URL for %s/%s (expires=%ds)",
                bucket_name,
                key,
                expires_in_seconds,
            )
            return url
        except Exception as e:
            raise StorageClientError(f"Signed URL generation failed: {e}") from e


# ---------------------------------------------------------------------------
# Azure Blob Storage implementation
# ---------------------------------------------------------------------------


class AzureBlobStorageClient(StorageClient):
    """StorageClient backed by the native Azure Blob Storage SDK."""

    def __init__(
        self, account_name: str, account_key: str, timeout_seconds: int = 30
    ):
        from azure.storage.blob import BlobServiceClient

        self._account_name = account_name
        self._account_key = account_key
        account_url = f"https://{account_name}.blob.core.windows.net"
        self._service_client = BlobServiceClient(
            account_url=account_url,
            credential=account_key,
            connection_timeout=timeout_seconds,
        )

    def check_bucket(self, bucket_name: str, create_if_missing: bool = True) -> None:
        try:
            container_client = self._service_client.get_container_client(bucket_name)
            if not container_client.exists():
                if create_if_missing:
                    self._create_container(bucket_name)
                else:
                    raise StorageClientError(
                        f"Container '{bucket_name}' does not exist"
                    )
            else:
                logger.debug("Container check passed: %s", bucket_name)
        except StorageClientError:
            raise
        except Exception as e:
            raise StorageClientError(f"Container check failed: {e}") from e

    def _create_container(self, bucket_name: str) -> None:
        """Create an Azure Blob container."""
        from azure.core.exceptions import ResourceExistsError

        try:
            container_client = self._service_client.get_container_client(bucket_name)
            container_client.create_container()
            logger.info("Created Azure container: %s", bucket_name)
        except ResourceExistsError:
            logger.debug("Container already exists (race condition): %s", bucket_name)
        except Exception as e:
            raise StorageClientError(
                f"Failed to create container '{bucket_name}': {e}"
            ) from e

    @_retry_transient
    def upload(
        self,
        bucket_name: str,
        key: str,
        data: bytes,
        content_type: str,
    ) -> None:
        from azure.storage.blob import ContentSettings

        try:
            blob_client = self._service_client.get_blob_client(
                container=bucket_name, blob=key
            )
            md5_digest = bytearray(hashlib.md5(data).digest())
            blob_client.upload_blob(
                data=data,
                overwrite=True,
                content_settings=ContentSettings(
                    content_type=content_type,
                    content_md5=md5_digest,
                ),
            )
            logger.debug("Uploaded %s/%s (%d bytes)", bucket_name, key, len(data))
        except Exception as e:
            raise StorageClientError(f"Upload failed: {e}") from e

    @_retry_transient
    def download(self, bucket_name: str, key: str) -> bytes:
        from azure.core.exceptions import ResourceNotFoundError

        try:
            blob_client = self._service_client.get_blob_client(
                container=bucket_name, blob=key
            )
            stream = blob_client.download_blob()
            data = stream.readall()
            logger.debug("Downloaded %s/%s (%d bytes)", bucket_name, key, len(data))
            return data
        except ResourceNotFoundError as e:
            raise BlobNotFoundError(f"Blob not found: {key}") from e
        except Exception as e:
            raise StorageClientError(f"Download failed: {e}") from e

    @_retry_transient
    def delete(self, bucket_name: str, key: str) -> None:
        """Idempotent delete — suppresses ResourceNotFoundError to match
        S3's idempotent behavior."""
        from azure.core.exceptions import ResourceNotFoundError

        try:
            blob_client = self._service_client.get_blob_client(
                container=bucket_name, blob=key
            )
            blob_client.delete_blob()
            logger.debug("Deleted %s/%s", bucket_name, key)
        except ResourceNotFoundError:
            logger.debug(
                "Delete no-op — blob already gone: %s/%s", bucket_name, key
            )
        except Exception as e:
            raise StorageClientError(f"Delete failed: {e}") from e

    @_retry_transient
    def generate_signed_url(
        self, bucket_name: str, key: str, expires_in_seconds: int
    ) -> str:
        from azure.storage.blob import BlobSasPermissions, generate_blob_sas

        try:
            now = datetime.datetime.now(datetime.timezone.utc)
            expiry = now + datetime.timedelta(seconds=expires_in_seconds)

            sas_token = generate_blob_sas(
                account_name=self._account_name,
                container_name=bucket_name,
                blob_name=key,
                account_key=self._account_key,
                permission=BlobSasPermissions(read=True),
                expiry=expiry,
                start=now,
            )

            blob_client = self._service_client.get_blob_client(
                container=bucket_name, blob=key
            )
            url = f"{blob_client.url}?{sas_token}"
            logger.debug(
                "Generated SAS URL for %s/%s (expires=%ds)",
                bucket_name,
                key,
                expires_in_seconds,
            )
            return url
        except Exception as e:
            raise StorageClientError(f"Signed URL generation failed: {e}") from e
