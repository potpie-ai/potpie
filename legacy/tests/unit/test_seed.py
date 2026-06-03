"""Unit tests for seed package (no database required)."""

import pytest

pytestmark = pytest.mark.unit


def test_deterministic_seed_uid_stable():
    from seed.uid import deterministic_seed_uid

    a = deterministic_seed_uid("default", "seed@example.com")
    b = deterministic_seed_uid("default", "seed@example.com")
    assert a == b
    assert len(a) == 36


def test_deterministic_seed_uid_email_case_insensitive():
    from seed.uid import deterministic_seed_uid

    assert deterministic_seed_uid("p", "A@B.COM") == deterministic_seed_uid("p", "a@b.com")


def test_validate_default_profile():
    from seed.apply import validate_profile

    assert validate_profile("default") == []


def test_repo_root_contains_pyproject():
    from seed.paths import repo_root

    assert (repo_root() / "pyproject.toml").is_file()


def test_manifest_model():
    from seed.paths import data_dir
    from seed.schemas import Manifest
    import json

    m = json.loads((data_dir("default") / "manifest.json").read_text())
    Manifest.model_validate(m)


def test_require_existing_user_uid_returns_uid():
    from unittest.mock import MagicMock

    from seed.apply import require_existing_user_uid

    row = MagicMock()
    row.uid = "firebase-uid-1"
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = row
    assert require_existing_user_uid(db, "a@b.com") == "firebase-uid-1"


def test_require_existing_user_uid_raises_when_missing():
    from unittest.mock import MagicMock

    from seed.apply import require_existing_user_uid

    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = None
    with pytest.raises(ValueError, match="No user with email"):
        require_existing_user_uid(db, "missing@example.com")
