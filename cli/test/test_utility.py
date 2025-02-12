import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from potpie.utility import Utility


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
