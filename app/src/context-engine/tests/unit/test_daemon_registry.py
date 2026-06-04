import pytest
from host.daemon_runtime.registry import Registry, UnknownPlugin


def test_register_and_create():
    reg: Registry[dict] = Registry()
    reg.register("greet", lambda **cfg: {"hello": cfg.get("name", "world")})
    out = reg.create("greet", name="alice")
    assert out == {"hello": "alice"}


def test_unknown_raises_with_known_names():
    reg: Registry[int] = Registry()
    reg.register("one", lambda: 1)
    with pytest.raises(UnknownPlugin) as ei:
        reg.create("two")
    assert "two" in str(ei.value)
    assert "one" in str(ei.value)


def test_duplicate_register_raises():
    reg: Registry[int] = Registry()
    reg.register("x", lambda: 1)
    with pytest.raises(ValueError):
        reg.register("x", lambda: 2)


def test_names_listed():
    reg: Registry[int] = Registry()
    reg.register("a", lambda: 1)
    reg.register("b", lambda: 2)
    assert sorted(reg.names()) == ["a", "b"]
