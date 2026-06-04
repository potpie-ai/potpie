import pytest
from pydantic import BaseModel
from domain.ports.daemon.operations import (
    OperationSpec, OperationRegistry, OperationError, OperationContext,
    AuthRequirement, OpKind,
)


class Echo(BaseModel):
    msg: str

class EchoOut(BaseModel):
    echoed: str


async def handler(inp: Echo, ctx: OperationContext) -> EchoOut:
    return EchoOut(echoed=inp.msg)


def test_register_and_lookup():
    reg = OperationRegistry()
    op = OperationSpec(
        name="echo.say", input_model=Echo, output_model=EchoOut,
        handler=handler, summary="echo", mutating=False,
    )
    reg.register(op)
    assert reg.get("echo.say") is op
    assert "echo.say" in [o.name for o in reg.all()]


def test_duplicate_name_rejected():
    reg = OperationRegistry()
    op = OperationSpec(name="dup", input_model=Echo, output_model=EchoOut, handler=handler, summary="")
    reg.register(op)
    with pytest.raises(ValueError):
        reg.register(op)


def test_operation_error_fields():
    err = OperationError(
        code="not_found", message="x", detail={"k": 1},
        recommended_next_action="try again",
    )
    assert err.code == "not_found"
    assert err.detail == {"k": 1}
    assert err.recommended_next_action == "try again"


def test_defaults():
    op = OperationSpec(name="d", input_model=Echo, output_model=EchoOut, handler=handler, summary="")
    assert op.mutating is False
    assert op.auth is AuthRequirement.REQUIRED
    assert op.kind is OpKind.UNARY
