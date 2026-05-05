from unittest.mock import MagicMock, patch

from app.modules.utils import gvisor_runner


def test_filter_safe_environment_variables_keeps_xdg_data_home(tmp_path):
    xdg_data_home = tmp_path / "xdg-data"
    filtered = gvisor_runner._filter_safe_environment_variables(
        {
            "PATH": "/usr/bin:/bin",
            "XDG_DATA_HOME": str(xdg_data_home),
        }
    )

    assert filtered["XDG_DATA_HOME"] == str(xdg_data_home)
