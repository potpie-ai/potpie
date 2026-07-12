"""Console entrypoint for the root-owned Potpie MCP process."""

from potpie.mcp.server import mcp
from potpie.runtime.logging import configure_logging


def main() -> None:
    configure_logging()
    mcp.run()


if __name__ == "__main__":
    main()


__all__ = ["main"]
