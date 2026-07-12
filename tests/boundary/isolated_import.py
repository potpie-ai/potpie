"""Subprocess harness for isolated package-import and wheel smoke tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True, slots=True)
class IsolatedPythonResult:
    returncode: int
    stdout: str
    stderr: str


class IsolatedPythonError(RuntimeError):
    """Raised when an isolated Python probe exits unsuccessfully."""

    def __init__(self, result: IsolatedPythonResult) -> None:
        super().__init__(
            "isolated Python probe failed "
            f"with exit code {result.returncode}: {result.stderr.strip()}"
        )
        self.result = result


def run_isolated_python(
    code: str,
    *,
    python: str | Path = sys.executable,
    import_roots: Sequence[str | Path] = (),
    cwd: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float = 30.0,
    check: bool = True,
) -> IsolatedPythonResult:
    """Run ``code`` with ``python -I`` and only explicit extra import roots.

    ``-I`` removes the current working directory and environment-provided
    Python paths. Explicit roots are inserted by the bootstrap so later tests
    can target an unpacked wheel or a temporary installation deliberately.
    """

    roots = [str(Path(root).resolve()) for root in import_roots]
    bootstrap = (
        "import sys\n"
        f"sys.path[:0] = {json.dumps(roots)}\n"
        f"exec(compile({code!r}, '<isolated-import-probe>', 'exec'))\n"
    )
    clean_env = dict(os.environ)
    clean_env.pop("PYTHONHOME", None)
    clean_env.pop("PYTHONPATH", None)
    clean_env.update(env or {})
    completed = subprocess.run(
        [str(python), "-I", "-c", bootstrap],
        cwd=str(cwd) if cwd is not None else None,
        env=clean_env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    result = IsolatedPythonResult(
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )
    if check and result.returncode != 0:
        raise IsolatedPythonError(result)
    return result


__all__ = [
    "IsolatedPythonError",
    "IsolatedPythonResult",
    "run_isolated_python",
]
