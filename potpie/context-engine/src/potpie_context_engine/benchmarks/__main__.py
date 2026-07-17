"""Allow ``python -m benchmarks ...``."""

from potpie_context_engine.benchmarks.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
