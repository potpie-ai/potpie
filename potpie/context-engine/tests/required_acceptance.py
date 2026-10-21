"""Opt-in pytest gate: a selected release scenario may not silently skip.

Load only for the curated supported-platform acceptance selection with
``-p tests.required_acceptance``. Broad suites retain unrelated optional tests.
"""

import pytest


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    skipped = reporter.stats.get("skipped", ()) if reporter is not None else ()
    if skipped:
        reporter.write_sep(
            "=", "Required acceptance scenarios skipped: release gate failed"
        )
        for report in skipped:
            reporter.write_line(report.nodeid)
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
