import pytest
import os
from pathlib import Path
from potpie.utility import Utility




@pytest.mark.parametrize(
    "test_params",
    [
        {
            "os_name": "posix",
            "home_path": "/home/testuser",
            "local_appdata": None,
            "expected_path": "/home/testuser/.config/potpie/potpie.pid",
        }
    ],
)
def test_pid_file_creation(monkeypatch, test_params):
    monkeypatch.setattr(os, "name", test_params["os_name"])

    def mock_home() -> Path:
        return Path(test_params["home_path"])

    monkeypatch.setattr(Path, "home", mock_home)

    def mock_getenv(key: str) -> str:
        if key == "LOCALAPPDATA":
            return test_params["local_appdata"]
        return ""

    monkeypatch.setattr(os, "getenv", mock_getenv)

    def mock_mkdir(*args, **kwargs):
        pass

    monkeypatch.setattr(Path, "mkdir", mock_mkdir)

    pid_file = Utility.create_path_of_pid_file()

    expected_path = Path(test_params["expected_path"]).as_posix()
    actual_path = Path(pid_file).as_posix()

    assert actual_path == expected_path


@pytest.mark.parametrize(
    "url, expected",
    [
        ("http://localhost:8001", True),
        ("http://localhost:8000", False),
        ("http://example.com", False),
        ("http://localhost:8001/api", False),
    ],
)
def test_base_url(url, expected):
    assert (Utility.base_url() == url) == expected


def test_get_user_id():
    assert Utility.get_user_id() == "defaultuser"


if __name__ == "__main__":
    pytest.main()
