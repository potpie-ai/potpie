from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from app.modules.media.media_controller import MediaController


@pytest.mark.asyncio
async def test_download_attachment_internal_success():
    controller = object.__new__(MediaController)
    controller.media_service = SimpleNamespace(
        get_attachment=AsyncMock(
            return_value=SimpleNamespace(
                file_name="diagram.png",
                mime_type="image/png",
            )
        ),
        get_attachment_data=AsyncMock(return_value=b"png-bytes"),
    )

    response = await controller.download_attachment_internal("att-123")

    assert response.status_code == 200
    assert response.media_type == "image/png"
    assert "Content-Disposition" in response.headers
    assert response.headers["Content-Length"] == str(len(b"png-bytes"))


@pytest.mark.asyncio
async def test_download_attachment_internal_not_found():
    controller = object.__new__(MediaController)
    controller.media_service = SimpleNamespace(
        get_attachment=AsyncMock(return_value=None),
        get_attachment_data=AsyncMock(return_value=b""),
    )

    with pytest.raises(HTTPException) as exc_info:
        await controller.download_attachment_internal("missing")
    assert exc_info.value.status_code == 404
