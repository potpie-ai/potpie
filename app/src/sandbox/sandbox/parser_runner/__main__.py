"""``potpie-parse <repo_dir>`` / ``python -m sandbox.parser_runner`` entry point.

Two callers reach this module: the Dockerfile installs a console-script
shim baked from this package, and the host ``[project.scripts]`` entry
in ``app/src/sandbox/pyproject.toml`` puts ``potpie-parse`` on PATH for
the local sandbox runtime. Both end up calling :func:`main`.
"""

from __future__ import annotations

import sys

from .runner import run


def main(argv: list[str] | None = None) -> int:
    """Process-script entry point.

    ``argv`` defaults to ``sys.argv[1:]`` so the
    ``[project.scripts] potpie-parse`` entry installed by the
    sandbox package can call ``main()`` with no arguments.
    """
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        sys.stderr.write(f"usage: {sys.argv[0]} <repo_dir>\n")
        return 2
    return run(args[0])


if __name__ == "__main__":
    raise SystemExit(main())
