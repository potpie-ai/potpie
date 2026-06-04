import pathlib, pytest
from host.daemon_runtime.ipc_auth import IpcAuthGate, AuthFailure
from domain.ports.daemon.operations import Principal


def test_token_match():
    g = IpcAuthGate(token="secret-1234")
    p = g.authenticate_token("secret-1234")
    assert isinstance(p, Principal)
    assert p.name == "local"


def test_token_mismatch():
    g = IpcAuthGate(token="secret-1234")
    with pytest.raises(AuthFailure):
        g.authenticate_token("wrong")


def test_uds_principal_from_peer_uid():
    g = IpcAuthGate(token=None)
    p = g.authenticate_uds(peer_uid=42)
    assert p.name == "local"
    assert p.attrs.get("uid") == 42


def test_generate_token_creates_file(tmp_path: pathlib.Path):
    f = tmp_path / "token"
    tok = IpcAuthGate.generate_token_file(f)
    assert f.read_text() == tok
    # owner-only permissions enforced
    assert oct(f.stat().st_mode & 0o777) == "0o600"


def test_token_configured_flag():
    assert IpcAuthGate(token="x").token_configured is True
    assert IpcAuthGate(token=None).token_configured is False
