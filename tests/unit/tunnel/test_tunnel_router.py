"""Unit tests for tunnel_router helpers."""
import pytest
from fastapi import HTTPException

from app.modules.tunnel.tunnel_router import _validate_workspace_id_hex


pytestmark = pytest.mark.unit


class TestValidateWorkspaceIdHex:
    def test_valid_16_hex_passes(self):
        _validate_workspace_id_hex("a" * 16)
        _validate_workspace_id_hex("0123456789abcdef")

    def test_invalid_length_raises(self):
        with pytest.raises(HTTPException) as exc_info:
            _validate_workspace_id_hex("abc")
        assert exc_info.value.status_code == 400
        assert "16 hex" in (exc_info.value.detail or "")

    def test_invalid_char_raises(self):
        with pytest.raises(HTTPException) as exc_info:
            _validate_workspace_id_hex("g" + "a" * 15)
        assert exc_info.value.status_code == 400
