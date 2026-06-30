"""Allow ``python -m benchmarks ...``."""

from benchmarks.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
