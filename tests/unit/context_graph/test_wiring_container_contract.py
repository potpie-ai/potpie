"""CGT-4: host ``build_container_*`` wiring contract (no Celery/Neo4j/LLM)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.modules.context_graph.wiring import (
    SqlalchemyPotResolution,
    UserScopedContextGraphPotResolution,
    _attach_agent_tools,
    build_container_for_session,
    build_container_for_user_session,
)

pytestmark = pytest.mark.unit


def _build_with_patches(*, user_id: str | None = None):
    db = MagicMock()
    agent = MagicMock()
    agent.add_extra_tools = MagicMock()
    jobs = MagicMock()
    connector_reg = MagicMock()
    github = MagicMock()
    github.kind.return_value = "github"
    linear = MagicMock()
    linear.kind.return_value = "linear"
    connector_reg.all.return_value = [github, linear]

    patches = [
        patch(
            "app.modules.context_graph.wiring.try_pydantic_deep_reconciliation_agent",
            return_value=agent,
        ),
        patch(
            "app.modules.context_graph.wiring.get_context_graph_job_queue",
            return_value=jobs,
        ),
        patch("app.modules.context_graph.wiring._attach_agent_tools"),
        patch(
            "app.modules.context_graph.wiring.CodeProviderFactory.create_provider_with_fallback",
            return_value=MagicMock(),
        ),
        patch(
            "app.modules.context_graph.wiring._build_connector_registry",
            return_value=connector_reg,
        ),
    ]

    started = [p.start() for p in patches]
    try:
        if user_id is None:
            container = build_container_for_session(db)
            pots_type = SqlalchemyPotResolution
        else:
            container = build_container_for_user_session(db, user_id)
            pots_type = UserScopedContextGraphPotResolution
        return container, db, agent, jobs, pots_type, connector_reg, started
    except Exception:
        for p in started:
            p.stop()
        raise


def _stop(patches: list) -> None:
    for p in patches:
        p.stop()


class TestContainerShape:
    def test_session_container_has_jobs_connectors_and_source_listing(self) -> None:
        container, _db, agent, jobs, pots_type, connector_reg, patches = (
            _build_with_patches()
        )
        try:
            assert container.jobs is jobs
            assert container.reconciliation_agent is agent
            assert container.pot_source_listing is not None
            assert isinstance(container.pots, pots_type)
            assert getattr(container.pots, "actor_scoped", False) is False
            assert container.connectors is connector_reg
            kinds = {c.kind() for c in connector_reg.all.return_value}
            assert kinds == {"github", "linear"}
        finally:
            _stop(patches)

    def test_user_session_container_uses_actor_scoped_resolver(self) -> None:
        container, _db, _agent, _jobs, pots_type, _reg, patches = _build_with_patches(
            user_id="user-99"
        )
        try:
            assert isinstance(container.pots, pots_type)
            assert container.pots.actor_scoped is True
            assert container.pots._user_id == "user-99"
        finally:
            _stop(patches)


class TestAttachAgentToolsBestEffort:
    def test_one_failing_surface_does_not_block_others(self) -> None:
        agent = MagicMock()
        captured: list = []

        def _capture(builders: list) -> None:
            captured.extend(builders)

        agent.add_extra_tools.side_effect = _capture
        db = MagicMock()

        github_builder = MagicMock(name="github")
        web_builder = MagicMock(name="web")

        with (
            patch(
                "app.modules.context_graph.wiring._sandbox_tools_disabled",
                return_value=False,
            ),
            patch(
                "adapters.outbound.agent_tools.sandbox.build_sandbox_tools",
                side_effect=RuntimeError("sandbox unavailable"),
            ),
            patch(
                "adapters.outbound.connectors.github.agent_tools.build_github_tools",
                return_value=github_builder,
            ),
            patch(
                "adapters.outbound.connectors.linear.agent_tools.build_linear_tools",
                side_effect=ImportError("linear optional"),
            ),
            patch(
                "app.modules.context_graph.agent_web_tools.build_web_tools",
                return_value=web_builder,
            ),
        ):
            _attach_agent_tools(agent, db, source_for_repo=lambda _r: MagicMock())

        assert agent.add_extra_tools.called
        assert github_builder in captured
        assert web_builder in captured

    def test_no_agent_is_a_noop(self) -> None:
        db = MagicMock()
        with patch(
            "adapters.outbound.agent_tools.sandbox.build_sandbox_tools",
        ) as build_sandbox:
            _attach_agent_tools(None, db, source_for_repo=lambda _r: MagicMock())
            build_sandbox.assert_not_called()
