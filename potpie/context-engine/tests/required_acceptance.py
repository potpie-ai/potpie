"""Partition Linux CI tests and require selected acceptance scenarios to run.

Loading this plugin without a tier retains the strict gate used by Windows CI.
"""

from pathlib import Path

import pytest


_TESTS_DIR = Path(__file__).resolve().parent
_GRAPH_FILES = frozenset(
    {
        "test_graph_reliability.py",
        "test_public_semantic_mutation_reliability.py",
        "test_record_capture_acceptance.py",
        "test_record_recovery.py",
        "test_temporal_backend_acceptance.py",
        "test_graph_journal.py",
        "test_falkordb_atomic.py",
        "test_falkordb_reliability.py",
        "test_falkordb_revision_integrity.py",
        "test_falkordb_write_entrypoints.py",
        "test_falkordb_roundtrip.py",
        "test_falkordb_claim_lifecycle.py",
        "test_falkordb_claim_corrections.py",
        "test_falkordb_temporal.py",
        "test_falkordb_verification.py",
        "test_neo4j_atomic_mutations.py",
        "test_neo4j_execution_reliability.py",
        "test_neo4j_revision_integrity.py",
        "test_neo4j_write_entrypoints.py",
        "test_neo4j_claim_lifecycle.py",
        "test_neo4j_claim_corrections.py",
        "test_neo4j_temporal_reads.py",
        "test_neo4j_verification.py",
        "test_scoped_claim_reads.py",
    }
)
_RESTART_SELECTORS = frozenset(
    {
        "test_falkordb_reliability.py::test_server_aof_receipt_survives_restart",
        "test_falkordb_reliability.py::test_lite_receipt_survives_process_restart",
        "test_neo4j_execution_reliability.py::test_native_server_restart_retains_execution_receipt",
        "test_journal_crash.py",
    }
)


def pytest_addoption(parser):
    parser.addoption(
        "--acceptance-tier",
        choices=("general", "graph", "restart"),
        help="Select one disjoint Linux CI tier; acceptance tiers may not skip.",
    )


def pytest_ignore_collect(collection_path, config):
    tier = config.getoption("acceptance_tier")
    if tier not in {"graph", "restart"} or collection_path.suffix != ".py":
        return None
    if not collection_path.is_relative_to(_TESTS_DIR):
        return None
    path = collection_path.name
    files = (
        _GRAPH_FILES
        if tier == "graph"
        else {selector.partition("::")[0] for selector in _RESTART_SELECTORS}
    )
    # Do not import unrelated optional modules: their module-level skips must
    # not fail an otherwise complete required acceptance selection.
    return path not in files


def _item_tier(item):
    path = item.path.name
    node = f"{path}::{item.nodeid.partition('::')[2]}"
    if path in _RESTART_SELECTORS or node in _RESTART_SELECTORS:
        return "restart"
    # Match the previous graph gate's `-k 'not restart'`. Other restart tests
    # remain in the general suite unless explicitly selected for serial CI.
    if path in _GRAPH_FILES and "restart" not in node.lower():
        return "graph"
    return "general"


def pytest_collection_modifyitems(config, items):
    tier = config.getoption("acceptance_tier")
    if tier is None:
        return
    selected, deselected = [], []
    for item in items:
        (selected if _item_tier(item) == tier else deselected).append(item)
    items[:] = selected
    config.hook.pytest_deselected(items=deselected)


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    if session.config.getoption("acceptance_tier") == "general":
        return
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    skipped = reporter.stats.get("skipped", ()) if reporter is not None else ()
    if skipped:
        reporter.write_sep(
            "=", "Required acceptance scenarios skipped: release gate failed"
        )
        for report in skipped:
            reporter.write_line(report.nodeid)
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
