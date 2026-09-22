"""W7 human rendering keeps meaningful details and omission guidance."""

from potpie.cli.read_presenter import (
    ReadPresentationContext,
    render_items_bullets,
    render_items_table,
)


class _Result:
    view = "debugging.prior_occurrences"
    backed = True
    unsupported = ()
    coverage = ()
    quality = {"status": "ok", "confidence": "high"}

    def to_dict(self):
        return {
            "view": self.view,
            "backed": self.backed,
            "unsupported": [],
            "coverage": [],
            "quality": self.quality,
        }


def _ctx() -> ReadPresentationContext:
    return ReadPresentationContext(
        view="debugging.prior_occurrences",
        detail="full",
        relations="summary",
        format_mode="bullets",
        sort="score",
        dedupe="auto",
        event_limit=1,
    )


def test_text_modes_render_details_and_omitted_item_guidance() -> None:
    items = [
        {
            "entity_key": "fix:pool",
            "entity_type": "Fix",
            "summary": "Close leaked connections",
            "details": {
                "root_cause": "connections leaked on cancellation",
                "fix_steps": ["close in finally"],
                "verification_outcomes": [
                    {"succeeded": False, "outcome": "failed"}
                ],
            },
        },
        {"entity_key": "fix:other", "entity_type": "Fix", "summary": "Other"},
    ]

    for output in (
        render_items_bullets(_Result(), items, _ctx()),
        render_items_table(items, _ctx(), result=_Result()),
    ):
        assert "root_cause: connections leaked on cancellation" in output
        assert '"succeeded": false' in output
        assert "omitted_items=1" in output
        assert "larger --limit" in output
