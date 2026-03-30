"""ingest_episode use case."""

from datetime import datetime, timezone
from unittest.mock import MagicMock

from application.use_cases.ingest_episode import ingest_episode


def test_ingest_episode_disabled():
    episodic = MagicMock()
    episodic.enabled = False
    out = ingest_episode(
        episodic,
        "p1",
        "n",
        "body",
        "src",
        datetime.now(timezone.utc),
    )
    assert out == {"episode_uuid": None}
    episodic.add_episode.assert_not_called()


def test_ingest_episode_ok():
    episodic = MagicMock()
    episodic.enabled = True
    episodic.add_episode.return_value = "uuid-1"
    t = datetime(2025, 1, 2, tzinfo=timezone.utc)
    out = ingest_episode(episodic, "p1", "n", "body", "src", t)
    assert out == {"episode_uuid": "uuid-1"}
    episodic.add_episode.assert_called_once_with(
        pot_id="p1",
        name="n",
        episode_body="body",
        source_description="src",
        reference_time=t,
    )
