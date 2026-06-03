"""Unit tests for MediaService (constants, errors, init)."""
from unittest.mock import MagicMock, patch

import pytest

from app.modules.media.media_service import MediaService, MediaServiceError
from app.modules.media.media_model import StorageProvider


pytestmark = pytest.mark.unit


class TestMediaServiceError:
    def test_media_service_error(self):
        err = MediaServiceError("test error")
        assert str(err) == "test error"


class TestMediaServiceConstants:
    def test_allowed_image_types(self):
        assert "image/jpeg" in MediaService.ALLOWED_IMAGE_TYPES
        assert MediaService.ALLOWED_IMAGE_TYPES["image/png"] == "PNG"

    def test_mime_to_extension(self):
        assert MediaService.MIME_TO_EXTENSION["image/jpeg"] == "jpg"
        assert MediaService.MIME_TO_EXTENSION["application/pdf"] == "pdf"

    def test_max_image_size(self):
        assert MediaService.MAX_IMAGE_SIZE == 10 * 1024 * 1024

    def test_max_dimension(self):
        assert MediaService.MAX_DIMENSION == 2048


class TestMediaServiceInit:
    @patch("app.modules.media.media_service.config_provider")
    def test_init_multimodal_disabled(self, mock_config):
        mock_config.get_is_multimodal_enabled.return_value = False
        service = MediaService(MagicMock())
        assert service.is_multimodal_enabled is False
        assert service.storage_provider == StorageProvider.LOCAL
        assert service.s3_client is None
