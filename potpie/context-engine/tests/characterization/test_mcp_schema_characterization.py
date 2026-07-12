from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from adapters.inbound.mcp.server import mcp

pytestmark = pytest.mark.unit

SCHEMA_FIXTURE = Path(__file__).parent / "fixtures" / "mcp_context_tool_arguments.json"
CHARACTERIZED_TOOLS = {
    "context_resolve",
    "context_search",
    "context_record",
}
SCHEMA_KEYS = ("type", "anyOf", "default")


def _argument_contract(schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": schema["type"],
        "required": list(schema.get("required", [])),
        "properties": {
            name: {
                key: property_schema[key]
                for key in SCHEMA_KEYS
                if key in property_schema
            }
            for name, property_schema in schema["properties"].items()
        },
    }


def test_existing_mcp_argument_schemas_match_characterization_fixture() -> None:
    expected = json.loads(SCHEMA_FIXTURE.read_text(encoding="utf-8"))
    tools = asyncio.run(mcp.list_tools())
    actual = {
        tool.name: _argument_contract(tool.inputSchema)
        for tool in tools
        if tool.name in CHARACTERIZED_TOOLS
    }

    assert actual == expected
