"""Engine-side regressions for the potpie 2.0.1 release-test findings.

These all concern *what an error tells the caller*: which pot it searched,
and which flag the caller must type. Each one was a case where the message
named an internal field and pointed at nothing the user could act on.
"""

from __future__ import annotations

import pytest

from potpie_context_engine.application.services.graph_service import (
    _missing_required_scope_message,
)
from potpie_context_engine.core.context_records import (
    ContextRecordValidationError,
    REQUIRED_RECORD_DETAILS,
    record_detail_choices,
    required_record_details,
    validate_record_payload,
)
from potpie_context_engine.core.errors import (
    CLI_FLAG_FOR_FIELD,
    missing_field_message,
)

pytestmark = pytest.mark.unit


class _ViewContract:
    def __init__(self, name, required_scope=(), required_any_scope=()):
        self.name = name
        self.required_scope = required_scope
        self.required_any_scope = required_any_scope


class TestMissingScopeMessageNamesTheFlag:
    """m6: neither `service` nor `anchor_entity_key` is a flag on graph read."""

    def test_any_of_requirements_spell_out_the_scope_syntax(self) -> None:
        message = _missing_required_scope_message(
            _ViewContract(
                "infra_topology.service_neighborhood",
                required_any_scope=("service", "anchor_entity_key"),
            )
        )

        assert "requires one of service, anchor_entity_key" in message
        assert "--scope service:<value>" in message
        assert "--scope anchor_entity_key:<value>" in message

    def test_keys_with_their_own_flag_are_named_by_that_flag(self) -> None:
        message = _missing_required_scope_message(
            _ViewContract(
                "knowledge.document_context",
                required_any_scope=("query", "environment", "source_ref", "repo"),
            )
        )

        assert "--query <text>" in message
        assert "--environment <name>" in message
        assert "--source-ref <ref>" in message
        assert "--repo <owner/repo>" in message

    def test_the_generic_scope_key_is_not_rendered_as_scope_scope(self) -> None:
        message = _missing_required_scope_message(
            _ViewContract("decisions.active_decisions", required_any_scope=("scope",))
        )

        assert "--scope <key>:<value>" in message
        assert "--scope scope:" not in message

    def test_all_of_requirements_are_spelled_out_too(self) -> None:
        message = _missing_required_scope_message(
            _ViewContract("x.y", required_scope=("service",))
        )

        assert "all of service" in message
        assert "--scope service:<value>" in message


class TestRequiredFieldMessagesNameTheFlag:
    def test_inbox_actor_fields_point_at_by(self) -> None:
        assert missing_field_message("claimed_by").endswith("(pass --by)")
        assert missing_field_message("closed_by").endswith("(pass --by)")

    def test_positional_arguments_are_described_as_arguments(self) -> None:
        assert "the ITEM_ID argument" in missing_field_message("item_id")
        assert "the PLAN_ID argument" in missing_field_message("plan_id")

    def test_unmapped_fields_keep_the_plain_message(self) -> None:
        assert missing_field_message("nonesuch") == "nonesuch is required"

    def test_every_mapped_flag_is_a_flag_or_a_named_argument(self) -> None:
        for field, flag in CLI_FLAG_FOR_FIELD.items():
            assert flag.startswith("--") or "argument" in flag, field


class TestRequiredRecordDetailsMatchTheValidators:
    """M5: five record types were unreachable because no flag set their detail.

    The published table must stay in lockstep with the validators, or the CLI
    will advertise a detail the engine does not want (or miss one it does).
    """

    @pytest.mark.parametrize("record_type", sorted(REQUIRED_RECORD_DETAILS))
    def test_omitting_a_required_detail_is_rejected(self, record_type: str) -> None:
        fields = required_record_details(record_type)
        assert fields
        for omitted in fields:
            details = {
                field: _legal_value(record_type, field)
                for field in fields
                if field != omitted
            }
            with pytest.raises(ContextRecordValidationError):
                validate_record_payload(
                    record_type=record_type, summary="s", details=details
                )

    @pytest.mark.parametrize("record_type", sorted(REQUIRED_RECORD_DETAILS))
    def test_supplying_every_required_detail_validates(self, record_type: str) -> None:
        details = {
            field: _legal_value(record_type, field)
            for field in required_record_details(record_type)
        }
        assert validate_record_payload(
            record_type=record_type, summary="s", details=details
        )

    def test_free_form_types_require_nothing(self) -> None:
        assert required_record_details("fix") == ()
        assert required_record_details("investigation") == ()

    def test_constrained_details_publish_their_vocabulary(self) -> None:
        assert record_detail_choices("verification", "outcome") == (
            "didnt_work",
            "partial",
            "worked",
        )
        assert record_detail_choices("decision", "rationale") == ()


def _legal_value(record_type: str, field: str) -> str:
    choices = record_detail_choices(record_type, field)
    return choices[0] if choices else "value"
