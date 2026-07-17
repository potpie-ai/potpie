"""Static logo ASCII asset."""

from potpie_context_engine.adapters.inbound.cli.ui.static_logo_loader import (
    load_static_logo,
    resolve_static_logo_path,
)


def test_static_logo_asset_exists() -> None:
    path = resolve_static_logo_path()
    assert path is not None
    assert path.name == "potpie-logo-static.json"


def test_static_logo_loads() -> None:
    load_static_logo.cache_clear()
    art = load_static_logo()
    assert art is not None
    assert art.width == 96
    assert art.height == 20
    assert art.chomp_token
    assert len(art.text.plain) > 100
