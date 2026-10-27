"""Real graph workbench fixture with storage isolated to one test."""

from dataclasses import dataclass
from pathlib import Path

import pytest

from potpie.cli.commands import _common
from potpie_context_engine.adapters.outbound.graph.backends.in_memory_backend import (
    InMemoryGraphBackend,
)
from potpie_context_engine.adapters.outbound.graph.inbox_stores.local_json import (
    LocalJsonGraphInboxStore,
)
from potpie_context_engine.adapters.outbound.graph.plan_stores.local_json import (
    LocalJsonGraphPlanStore,
)
from potpie_context_engine.core.runtime import GraphRuntime, build_graph_runtime
from tests.graph_cli_support import _Graph, _Host


@dataclass
class GraphWorkbenchFixture:
    home: Path
    backend: InMemoryGraphBackend
    runtime: GraphRuntime

    @property
    def plans(self) -> LocalJsonGraphPlanStore:
        return LocalJsonGraphPlanStore(home=self.home)

    @property
    def inbox(self) -> LocalJsonGraphInboxStore:
        return LocalJsonGraphInboxStore(home=self.home)


@pytest.fixture(name="graph_workbench_runtime")
def _graph_workbench_runtime(tmp_path):
    home = tmp_path / "workbench"
    backend = InMemoryGraphBackend()
    runtime = build_graph_runtime(
        backend,
        LocalJsonGraphPlanStore(home=home),
        LocalJsonGraphInboxStore(home=home),
    )
    previous_runtime = _common._state["runtime"]
    previous_json = _common.is_json()
    _common.set_runtime(_Host(_Graph(), graph_workbench=runtime.workbench))
    _common.set_json(True)
    try:
        yield GraphWorkbenchFixture(home, backend, runtime)
    finally:
        _common.set_runtime(previous_runtime)
        _common.set_json(previous_json)
