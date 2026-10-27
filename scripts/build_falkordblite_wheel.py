#!/usr/bin/env python3
"""Prefetch immutable native inputs or build the repaired wheel with PEP517."""

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("dist/falkordblite"))
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    project = Path(__file__).resolve().parent / "packaging" / "falkordblite"
    spec = importlib.util.spec_from_file_location(
        "potpie_falkordblite_build", project / "_build.py"
    )
    backend = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(backend)
    cache = args.cache_dir.resolve()
    backend.fetch_sources(cache)
    if not args.download_only:
        env = dict(
            os.environ, POTPIE_NATIVE_SOURCE_CACHE=str(cache), POTPIE_NATIVE_OFFLINE="1"
        )
        subprocess.run(  # noqa: S603 - fixed interpreter and build module
            [
                sys.executable,
                "-m",
                "build",
                "--wheel",
                "--outdir",
                str(args.out_dir.resolve()),
                str(project),
            ],
            env=env,
            check=True,
        )


if __name__ == "__main__":
    main()
