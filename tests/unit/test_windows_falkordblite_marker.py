import tomllib
from pathlib import Path


def test_falkordblite_dep_is_skipped_on_windows() -> None:
    pyproject_path = Path(__file__).resolve().parents[2] / "potpie/context-engine/pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

    local_extra = pyproject["project"]["optional-dependencies"]["local"]
    falkordblite_dep = next(dep for dep in local_extra if dep.startswith("falkordblite"))

    assert "platform_system != 'Windows'" in falkordblite_dep
