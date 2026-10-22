"""Run native ARM acceptance against installed wheels, outside the checkout.

Invoke with the clean environment's Python (preferably ``python -I``). Only
tests are copied into the temporary working directory; product imports must
come from this interpreter's site-packages. A selected test may not skip.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import xml.etree.ElementTree as ET


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def verify_installation(target: str) -> None:
    tag = sysconfig.get_platform().lower().replace("_", "-")
    require(platform.machine().lower() in {"arm64", "aarch64"}, "ARM host required")
    require(
        (target == "windows" and tag == "win-arm64")
        or (target == "linux" and tag == "linux-aarch64"),
        f"Native {target} ARM interpreter required, found {tag}",
    )
    sites = {
        Path(sysconfig.get_path(name)).resolve() for name in ("purelib", "platlib")
    }
    origins = {}
    for name in ("potpie", "potpie_context_engine"):
        module = importlib.import_module(name)
        origin = Path(module.__file__).resolve()
        require(
            any(origin.is_relative_to(site) for site in sites),
            f"Source import: {origin}",
        )
        origins[name] = str(origin)
    if target == "linux":
        require(
            importlib.metadata.version("potpie-falkordblite") == "0.10.0.post1",
            "Corrected FalkorDBLite distribution was not installed",
        )
        try:
            importlib.metadata.distribution("falkordblite")
        except importlib.metadata.PackageNotFoundError:
            pass
        else:
            raise RuntimeError("Original and corrected redislite distributions overlap")
    else:
        require(
            importlib.util.find_spec("sqlite_vec") is None,
            "Unexpected sqlite-vec dependency",
        )
        require(
            importlib.util.find_spec("grpc") is None,
            "Unexpected native gRPC dependency",
        )
        require(
            importlib.util.find_spec("opentelemetry.exporter.otlp.proto.http")
            is not None,
            "HTTP telemetry transport was not installed",
        )
    print(
        json.dumps(
            {"platform": tag, "python": sys.version, "origins": origins}, indent=2
        )
    )


def run_suite(directory: Path, report: Path, selections: list[str]) -> None:
    subprocess.run(  # noqa: S603 - fixed pytest executable and curated test selections.
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--tb=short",
            "--import-mode=importlib",
            "-p",
            "no:cacheprovider",
            f"--junitxml={report}",
            *selections,
        ],
        cwd=directory,
        check=True,
    )
    # This report was just produced by the local pytest subprocess above.
    cases = ET.parse(report).getroot().findall(".//testcase")  # noqa: S314
    require(bool(cases), f"No acceptance tests ran: {report}")
    skipped = [
        case.attrib.get("name") for case in cases if case.find("skipped") is not None
    ]
    require(not skipped, f"Required acceptance tests skipped: {skipped}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--target", required=True, choices=("windows", "linux"))
    parser.add_argument("--reports", required=True, type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    reports = args.reports.resolve()
    reports.mkdir(parents=True, exist_ok=True)
    # Never inherit a checkout import path or a real user's local state.
    os.environ.pop("PYTHONPATH", None)
    os.environ.pop("PYTEST_ADDOPTS", None)
    os.environ.update(
        {
            "POTPIE_TELEMETRY_DISABLED": "1",
            "POTPIE_ENVIRONMENT": "production",
            "CONTEXT_ENGINE_EMBEDDER": "hashing",
            "PYTHON_KEYRING_BACKEND": "keyring.backends.null.Keyring",
        }
    )
    with tempfile.TemporaryDirectory(prefix="potpie-arm-") as temporary:
        work = Path(temporary)
        os.chdir(work)
        os.environ["CONTEXT_ENGINE_HOME"] = str(work / "home")
        os.environ["POTPIE_HARNESS_HOME"] = str(work / "harness")
        verify_installation(args.target)
        for name, source in (
            ("root", repo),
            ("engine", repo / "potpie/context-engine"),
        ):
            destination = work / name
            shutil.copytree(
                source / "tests",
                destination / "tests",
                ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"),
            )
            (destination / "pytest.ini").write_text(
                "[pytest]\nasyncio_mode = auto\nmarkers =\n    integration: native acceptance\n",
                encoding="utf-8",
            )
        # Qdrant is a separate service. Linux exercises the real service;
        # Windows exercises its REST adapter without requiring Docker or
        # changing the native Ladybug graph/daemon acceptance.
        resource_tests = [
            "tests/unit/test_qdrant_http.py",
            "tests/unit/test_qdrant_resource_sync.py",
        ]
        if args.target == "linux":
            os.environ["POTPIE_REQUIRE_QDRANT"] = "1"
            resource_tests += [
                "tests/conformance/test_resource_index_conformance.py",
                "tests/conformance/test_resource_revision_acceptance.py",
                "tests/unit/test_resource_index_drain.py",
                "tests/unit/test_ranking.py",
                "tests/integration/test_qdrant_resource_index.py",
            ]
        run_suite(work / "engine", reports / "resources.xml", resource_tests)
        transport_tests = ["tests/unit/test_otel_transport.py"]
        if args.target == "windows":
            # This install deliberately contains only the HTTP exporter.
            # The gRPC default is covered on Linux and in the unit suite.
            transport_tests += ["-k", "not other_platforms_keep_grpc_port_4317"]
        run_suite(work / "engine", reports / "telemetry.xml", transport_tests)
        if args.target == "windows":
            os.environ["POTPIE_REQUIRE_LADYBUG"] = "1"
            run_suite(
                work / "root",
                reports / "daemon.xml",
                [
                    "tests/integration/test_ladybug_runtime.py",
                ],
            )
        else:
            run_suite(
                work / "root",
                reports / "runtime.xml",
                [
                    "tests/unit/test_runtime_composition.py::test_local_runtime_accepts_backend_env",
                ],
            )
            run_suite(
                work / "engine",
                reports / "reliability.xml",
                [
                    "tests/integration/test_falkordb_reliability.py",
                    "-k",
                    "not server",
                ],
            )
            run_suite(
                work / "engine",
                reports / "lifecycle-snapshot.xml",
                [
                    "tests/integration/test_falkordb_lite_server_lifecycle.py",
                    "tests/integration/test_falkordb_snapshot.py",
                ],
            )


if __name__ == "__main__":
    main()
