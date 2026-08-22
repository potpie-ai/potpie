"""Verify installed distribution boundaries from outside the source packages."""

from __future__ import annotations

import argparse
import importlib.util


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expect-root", action="store_true")
    args = parser.parse_args()

    assert importlib.util.find_spec("potpie_context_core") is None
    assert importlib.util.find_spec("potpie_context_engine") is not None
    __import__("potpie_context_engine")

    root_spec = importlib.util.find_spec("potpie")
    if args.expect_root:
        assert root_spec is not None
        __import__("potpie")
    else:
        assert root_spec is None


if __name__ == "__main__":
    main()
