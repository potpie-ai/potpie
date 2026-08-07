"""Mypy smoke target for the installed public context distributions."""

from __future__ import annotations

from typing import Mapping

from potpie_context_core.api import GraphDefinition
from potpie_context_engine.api import Candidate


def definition_metadata(definition: GraphDefinition) -> Mapping[str, object]:
    return definition.status_metadata()


def candidate_key(candidate: Candidate) -> str:
    return candidate.candidate_key
