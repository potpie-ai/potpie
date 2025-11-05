import os
from abc import ABC, abstractmethod
from typing import Any, Dict

from botocore.config import Config


class StorageBackendStrategy(ABC):
    """Strategy interface for storage backend configuration"""

    @abstractmethod
    def get_descriptor(self, config_provider) -> Dict[str, Any]:
        """Get storage configuration descriptor for this backend"""
        pass

    @abstractmethod
    def is_ready(self, config_provider) -> bool:
        """Check if this storage backend is properly configured"""
        pass


class S3StorageStrategy(StorageBackendStrategy):
    """AWS S3 storage strategy"""

    def get_descriptor(self, config_provider) -> Dict[str, Any]:
        return {
            "provider": "s3",
            "bucket_name": config_provider.s3_bucket_name,
            "client_kwargs": {
                "aws_access_key_id": config_provider.aws_access_key,
                "aws_secret_access_key": config_provider.aws_secret_key,
                "region_name": config_provider.aws_region,
            },
        }

    def is_ready(self, config_provider) -> bool:
        return all(
            [
                config_provider.s3_bucket_name,
                config_provider.aws_region,
                config_provider.aws_access_key,
                config_provider.aws_secret_key,
            ]
        )


class GCSStorageStrategy(StorageBackendStrategy):
    """Google Cloud Storage strategy with S3 interoperability"""

    def get_descriptor(self, config_provider) -> Dict[str, Any]:
        gcs_access_key = os.getenv("GCS_HMAC_ACCESS_KEY")
        gcs_secret_key = os.getenv("GCS_HMAC_SECRET_KEY")

        if not gcs_access_key or not gcs_secret_key:
            raise ValueError(
                "GCS_HMAC_ACCESS_KEY and GCS_HMAC_SECRET_KEY must be configured for GCS backend"
            )

        return {
            "provider": "gcs",
            "bucket_name": config_provider.gcp_bucket_name,
            "client_kwargs": {
                "aws_access_key_id": gcs_access_key,
                "aws_secret_access_key": gcs_secret_key,
                "endpoint_url": "https://storage.googleapis.com",
                "config": Config(
                    signature_version="s3",
                    s3={"addressing_style": "path", "payload_signing_enabled": False},
                ),
            },
        }

    def is_ready(self, config_provider) -> bool:
        return all(
            [
                config_provider.gcp_project_id,
                config_provider.gcp_bucket_name,
                os.getenv("GCS_HMAC_ACCESS_KEY"),
                os.getenv("GCS_HMAC_SECRET_KEY"),
            ]
        )


class AzureStorageStrategy(StorageBackendStrategy):
    """Azure Blob Storage strategy (future implementation)"""

    def get_descriptor(self, config_provider) -> Dict[str, Any]:
        azure_account_name = os.getenv("AZURE_ACCOUNT_NAME")
        azure_account_key = os.getenv("AZURE_ACCOUNT_KEY")
        azure_container_name = os.getenv("AZURE_CONTAINER_NAME")

        if not all([azure_account_name, azure_account_key, azure_container_name]):
            raise ValueError(
                "AZURE_ACCOUNT_NAME, AZURE_ACCOUNT_KEY, and AZURE_CONTAINER_NAME must be configured for Azure backend"
            )

        return {
            "provider": "azure",
            "bucket_name": azure_container_name,
            "client_kwargs": {
                "aws_access_key_id": azure_account_name,
                "aws_secret_access_key": azure_account_key,
                "endpoint_url": f"https://{azure_account_name}.blob.core.windows.net",
            },
        }

    def is_ready(self, config_provider) -> bool:
        return all(
            [
                os.getenv("AZURE_ACCOUNT_NAME"),
                os.getenv("AZURE_ACCOUNT_KEY"),
                os.getenv("AZURE_CONTAINER_NAME"),
            ]
        )
