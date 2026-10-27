"""Keep native distribution repair reproducible and platform-correct."""

# ruff: noqa: S101 - pytest assertions are the test oracle.

import hashlib
import importlib.util
import io
import struct
import tarfile
import zipfile
from email.parser import Parser
from pathlib import Path
from types import SimpleNamespace

from packaging.tags import Tag, parse_tag

import pytest

PROJECT = Path(__file__).resolve().parents[2] / "scripts" / "packaging" / "falkordblite"
spec = importlib.util.spec_from_file_location(
    "falkordblite_build", PROJECT / "_build.py"
)
backend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend)


def elf(machine=183):
    result = bytearray(64)
    result[:6] = b"\x7fELF\x02\x01"
    struct.pack_into("<H", result, 18, machine)
    return bytes(result)


def test_metadata_is_portable_and_needs_no_network(tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("metadata resolution must not download or compile")

    monkeypatch.setattr(backend.urllib.request, "urlopen", unexpected)
    monkeypatch.setattr(backend.subprocess, "run", unexpected)
    name = backend.prepare_metadata_for_build_wheel(tmp_path)
    metadata = (tmp_path / name / "METADATA").read_text()
    assert "Name: potpie-falkordblite" in metadata
    assert "Version: 0.10.0.post1" in metadata
    assert "Requires-Dist: redis>=7.1,<8" in metadata
    assert "Requires-Dist: falkordb>=1.6.1,<1.8" in metadata
    wheel_bytes = (tmp_path / name / "WHEEL").read_bytes()
    assert b"\r" not in wheel_bytes
    wheel = Parser().parsestr(wheel_bytes.decode())
    assert wheel["Wheel-Version"] == "1.0"
    assert wheel["Root-Is-Purelib"] == "false"
    tag = next(backend.sys_tags())
    assert parse_tag(wheel["Tag"]) == {
        Tag(tag.interpreter, tag.abi, "manylinux_2_36_aarch64")
    }


@pytest.mark.parametrize("prepared", [False, True])
def test_repaired_wheel_uses_portable_metadata(tmp_path, monkeypatch, prepared):
    """Repacking after auditwheel must preserve the frontend's metadata contract."""
    metadata_parent = tmp_path / "metadata"
    name = backend.prepare_metadata_for_build_wheel(metadata_parent)
    metadata = metadata_parent / name
    if prepared:
        (metadata / "build-decision.json").write_text('{"preserved": true}')
    expected = {path.name: path.read_bytes() for path in metadata.iterdir()}
    tag = Parser().parsestr(expected["WHEEL"].decode())["Tag"]
    wheel_path = tmp_path / f"{backend.DIST}-{backend.VERSION}-{tag}.whl"
    wheel_path.touch()
    packed = {}

    def run(command, *, check):
        assert check
        if command[3] == "unpack":
            unpacked = Path(command[-1]) / f"{backend.DIST}-{backend.VERSION}"
            (unpacked / name).mkdir(parents=True)
            (unpacked / "redislite").mkdir()
            (unpacked / name / "METADATA").write_text("old upstream metadata")
            (unpacked / name / "WHEEL").write_text("old auditwheel metadata")
            (unpacked / name / "licenses").mkdir()
            (unpacked / name / "licenses/LICENSE").write_text("upstream license")
        else:
            assert command[3] == "pack"
            unpacked = Path(command[4])
            for path in (unpacked / name).iterdir():
                if path.is_file():
                    packed[path.name] = path.read_bytes()
            assert (
                unpacked / name / "licenses/LICENSE"
            ).read_text() == "upstream license"
            assert (unpacked / "redislite/potpie_build.json").is_file()

    monkeypatch.setattr(backend.subprocess, "run", run)
    backend._write_metadata(wheel_path, metadata if prepared else None)
    assert packed == expected


def test_build_normalizes_metadata_after_native_repair(tmp_path, monkeypatch):
    metadata_parent = tmp_path / "metadata"
    metadata = metadata_parent / backend.prepare_metadata_for_build_wheel(
        metadata_parent
    )
    tag = Parser().parsestr((metadata / "WHEEL").read_text())["Tag"]
    name = f"{backend.DIST}-{backend.VERSION}-{tag}.whl"
    monkeypatch.setattr(
        backend,
        "sys",
        SimpleNamespace(platform="linux", executable=backend.sys.executable),
    )
    monkeypatch.setattr(backend, "platform", SimpleNamespace(machine=lambda: "aarch64"))
    monkeypatch.setattr(
        backend,
        "fetch_sources",
        lambda cache: {backend.SDIST: None, backend.MODULE: None},
    )
    monkeypatch.setattr(backend, "_prepare_source", lambda *args: tmp_path)

    def run(command, **kwargs):
        if "bdist_wheel" in command:
            output = Path(command[-1])
        else:
            assert command[2:4] == ["auditwheel", "repair"]
            assert command[command.index("--plat") + 1] == "manylinux_2_36_aarch64"
            assert "--only-plat" in command
            output = Path(command[command.index("--wheel-dir") + 1])
            output.mkdir()
        with zipfile.ZipFile(output / name, "w") as archive:
            archive.writestr("redislite/bin/redis-server", elf())
            archive.writestr("redislite/bin/falkordb.so", elf())

    normalized = []

    def write_metadata(wheel, prepared_metadata):
        assert wheel.parent.name == "repaired"
        assert prepared_metadata == metadata
        normalized.append(wheel.name)

    monkeypatch.setattr(backend.subprocess, "run", run)
    monkeypatch.setattr(backend, "_write_metadata", write_metadata)
    output = tmp_path / "output"
    assert backend.build_wheel(output, metadata_directory=metadata) == name
    assert normalized == [name]
    assert (output / name).is_file()


def test_source_distribution_contains_only_recipe(tmp_path):
    filename = backend.build_sdist(tmp_path)
    with tarfile.open(tmp_path / filename) as archive:
        names = {Path(name).name for name in archive.getnames()}
        assert names == {"pyproject.toml", "_build.py", "README.md", "PKG-INFO"}
        assert all(
            not member.issym() and not member.islnk() for member in archive.getmembers()
        )


@pytest.mark.parametrize(
    "data", [b"", b"not ELF", elf(62), b"\x7fELF\x01\x01" + b"\0" * 58]
)
def test_rejects_wrong_or_invalid_native_binary(data):
    with pytest.raises(RuntimeError):
        backend.require_arm64_elf(data, "module")


def test_accepts_arm64_elf():
    backend.require_arm64_elf(elf(), "module")


def test_cached_sources_are_verified_even_offline(tmp_path, monkeypatch):
    monkeypatch.setattr(
        backend,
        "SOURCES",
        {
            "input": (
                "https://example.invalid/source",
                hashlib.sha256(b"good").hexdigest(),
            )
        },
    )
    monkeypatch.setenv("POTPIE_NATIVE_OFFLINE", "1")
    with pytest.raises(RuntimeError, match="offline native build is missing"):
        backend.fetch_sources(tmp_path)
    (tmp_path / "input").write_bytes(b"bad")
    with pytest.raises(RuntimeError, match="SHA256 mismatch for cached"):
        backend.fetch_sources(tmp_path)
    (tmp_path / "input").write_bytes(b"good")
    assert backend.fetch_sources(tmp_path) == {"input": tmp_path / "input"}


def test_failed_download_does_not_publish_or_leave_partial_input(tmp_path, monkeypatch):
    monkeypatch.delenv("POTPIE_NATIVE_OFFLINE", raising=False)
    monkeypatch.setattr(
        backend,
        "SOURCES",
        {
            "input": (
                "https://example.invalid/source",
                hashlib.sha256(b"good").hexdigest(),
            )
        },
    )
    monkeypatch.setattr(
        backend.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(b"wrong")
    )
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        backend.fetch_sources(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_download_uses_verified_input(tmp_path, monkeypatch):
    monkeypatch.delenv("POTPIE_NATIVE_OFFLINE", raising=False)
    monkeypatch.setattr(
        backend,
        "SOURCES",
        {
            "input": (
                "https://example.invalid/source",
                hashlib.sha256(b"good").hexdigest(),
            )
        },
    )
    monkeypatch.setattr(
        backend.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(b"good")
    )
    assert backend.fetch_sources(tmp_path)["input"].read_bytes() == b"good"


def test_preparation_discards_builder_binaries_and_preserves_source(tmp_path):
    archive_path = tmp_path / "upstream.tar.gz"
    files = {
        "falkordb.so": elf(62),
        "redislite/bin/redis-server": elf(62),
        "redislite/bin/falkordb.so": elf(62),
        "redislite/__init__.py": b"# upstream implementation\n",
        "falkordblite.egg-info/PKG-INFO": b"stale metadata",
        "PKG-INFO": b"stale metadata",
        "setup.cfg": b"[metadata]\nname=falkordblite\nversion=0.10.0\n[options]\npackages=redislite\n",
        "MANIFEST.in": b"include falkordb.so\nrecursive-include redislite/bin *\n",
    }
    with tarfile.open(archive_path, "w:gz") as archive:
        for name, data in files.items():
            info = tarfile.TarInfo("falkordblite-0.10.0/" + name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    module = tmp_path / "native.so"
    module.write_bytes(elf())
    source = backend._prepare_source(archive_path, module, tmp_path / "source")
    assert (source / "falkordb.so").read_bytes() == elf()
    assert not (source / "redislite/bin").exists()
    assert not (source / "falkordblite.egg-info").exists()
    assert not (source / "PKG-INFO").exists()
    assert (source / "redislite/__init__.py").read_bytes() == files[
        "redislite/__init__.py"
    ]
    config = (source / "setup.cfg").read_text()
    assert "name = potpie-falkordblite" in config
    assert "redis>=7.1,<8" in config
    assert (
        source / "MANIFEST.in"
    ).read_text() == "exclude falkordb.so\nprune redislite/bin\n"


def test_wheel_validation_checks_both_server_and_module(tmp_path):
    wheel = tmp_path / "test.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("redislite/bin/falkordb.so", elf())
        archive.writestr("redislite/bin/redis-server", elf(62))
    with pytest.raises(RuntimeError, match="redis-server has ELF machine 62"):
        backend._verify_wheel(wheel)


def test_native_build_rejects_wrong_host_before_downloading(tmp_path, monkeypatch):
    monkeypatch.setattr(backend.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(
        backend, "fetch_sources", lambda *args: pytest.fail("unexpected fetch")
    )
    with pytest.raises(RuntimeError, match="native Linux ARM64"):
        backend.build_wheel(tmp_path)


pytestmark = [pytest.mark.source_contract]
