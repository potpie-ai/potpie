from mcp.server.fastmcp import FastMCP

mcp = FastMCP("context-engine")


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
