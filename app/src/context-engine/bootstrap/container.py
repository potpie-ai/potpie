"""Compose adapters and ports (dependency injection)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from sqlalchemy.orm import Session

from adapters.outbound.github.source_control import PyGithubSourceControl
from adapters.outbound.graphiti.episodic import GraphitiEpisodicAdapter
from adapters.outbound.neo4j.structural import Neo4jStructuralAdapter
from adapters.outbound.postgres.ledger import SqlAlchemyIngestionLedger
from adapters.outbound.settings_env import EnvContextEngineSettings
from domain.ports.episodic_graph import EpisodicGraphPort
from domain.ports.pot_resolution import PotResolutionPort
from domain.ports.settings import ContextEngineSettingsPort
from domain.ports.source_control import SourceControlPort
from domain.ports.structural_graph import StructuralGraphPort


@dataclass
class ContextEngineContainer:
    """Wired dependencies for use cases."""

    settings: ContextEngineSettingsPort
    episodic: EpisodicGraphPort
    structural: StructuralGraphPort
    pots: PotResolutionPort
    """Given repo_name (e.g. owner/repo), return API client for that repo."""
    source_for_repo: Callable[[str], SourceControlPort]

    def ledger(self, session: Session) -> SqlAlchemyIngestionLedger:
        return SqlAlchemyIngestionLedger(session)


def build_container(
    *,
    settings: ContextEngineSettingsPort | None = None,
    pots: PotResolutionPort,
    source_for_repo: Callable[[str], SourceControlPort],
) -> ContextEngineContainer:
    s = settings or EnvContextEngineSettings()
    return ContextEngineContainer(
        settings=s,
        episodic=GraphitiEpisodicAdapter(s),
        structural=Neo4jStructuralAdapter(s),
        pots=pots,
        source_for_repo=source_for_repo,
    )


def build_container_with_github_token(
    *,
    token: str,
    pots: PotResolutionPort,
    settings: ContextEngineSettingsPort | None = None,
) -> ContextEngineContainer:
    try:
        from github import Auth, Github

        gh = Github(auth=Auth.Token(token))
    except ImportError:
        from github import Github

        gh = Github(token)

    def source_for_repo(_repo_name: str) -> SourceControlPort:
        return PyGithubSourceControl(gh)

    return build_container(settings=settings, pots=pots, source_for_repo=source_for_repo)
