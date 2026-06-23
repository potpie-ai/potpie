"""Allow ``python -m context_engine.benchmarks ...``."""

from context_engine.benchmarks.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
