"""Build the Linux ARM64 repair from immutable upstream inputs.

Only the build downloads source/binaries. The installed package contains the
ordinary upstream redislite implementation and an architecture-correct module;
it never downloads or replaces native code at runtime.
"""

from __future__ import annotations

import configparser
import hashlib
import io
import json
import os
import platform
import shutil
import struct
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import urllib.request
import zipfile
from pathlib import Path

from packaging.tags import sys_tags

ROOT = Path(__file__).resolve().parent
DIST = "potpie_falkordblite"
VERSION = "0.10.0.post1"
DIST_INFO = f"{DIST}-{VERSION}.dist-info"
WHEEL_PLATFORM = "manylinux_2_36_aarch64"
SDIST = "falkordblite-0.10.0.tar.gz"
MODULE = "falkordb-arm64v8-v4.18.3.so"
SOURCES = {
    SDIST: (
        "https://files.pythonhosted.org/packages/70/03/afbb6e0f03c302aa9b64a38e1a2c43664f86921f0dcbf03ec9e31da06ac6/falkordblite-0.10.0.tar.gz",
        "65a72abafd30711f699c15571df6959edb8901605053ce940ccdd837832e709b",
    ),
    MODULE: (
        "https://github.com/FalkorDB/FalkorDB/releases/download/v4.18.3/falkordb-arm64v8.so",
        "11a71ded30e7a492e2c30592fddaebe119890224de09d9a931d86a4f564a34d6",
    ),
}


def _project() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]


def _metadata() -> bytes:
    project = _project()
    headers = [
        "Metadata-Version: 2.3",
        f"Name: {project['name']}",
        f"Version: {project['version']}",
        f"Summary: {project['description']}",
        f"Requires-Python: {project['requires-python']}",
        "Description-Content-Type: text/markdown",
    ]
    headers.extend(f"Requires-Dist: {dep}" for dep in project["dependencies"])
    headers.extend(
        f"Project-URL: {key}, {value}" for key, value in project["urls"].items()
    )
    return ("\n".join(headers) + "\n\n" + (ROOT / "README.md").read_text()).encode()


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    """Resolve the workspace on any OS without fetching or compiling anything."""
    target = Path(metadata_directory) / DIST_INFO
    target.mkdir(parents=True, exist_ok=True)
    (target / "METADATA").write_bytes(_metadata())
    # Upstream declares an extension, so retain this interpreter's ABI while
    # describing the native Linux ARM64 target even when resolving on Windows.
    tag = next(sys_tags())
    (target / "WHEEL").write_text(
        "Wheel-Version: 1.0\n"
        "Generator: potpie-falkordblite\n"
        "Root-Is-Purelib: false\n"
        f"Tag: {tag.interpreter}-{tag.abi}-{WHEEL_PLATFORM}\n",
        encoding="utf-8",
        newline="\n",
    )
    return DIST_INFO


def get_requires_for_build_wheel(config_settings=None):
    return []


def get_requires_for_build_sdist(config_settings=None):
    return []


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fetch_sources(cache: Path) -> dict[str, Path]:
    """Populate a cache with verified inputs; never trust a pre-existing file."""
    cache.mkdir(parents=True, exist_ok=True)
    result = {}
    for name, (url, expected) in SOURCES.items():
        target = cache / name
        if not target.exists():
            if os.environ.get("POTPIE_NATIVE_OFFLINE") == "1":
                raise RuntimeError(f"offline native build is missing {target}")
            # Publish to the cache only after a complete, verified download.
            pending = None
            try:
                with tempfile.NamedTemporaryFile(dir=cache, delete=False) as stream:
                    pending = Path(stream.name)
                    with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310 - fixed HTTPS manifest
                        shutil.copyfileobj(response, stream)
                if _digest(pending) != expected:
                    raise RuntimeError(f"SHA256 mismatch for {name}")
                pending.replace(target)
            finally:
                if pending is not None:
                    pending.unlink(missing_ok=True)
        if _digest(target) != expected:
            raise RuntimeError(
                f"SHA256 mismatch for cached {name}; remove the invalid cache file"
            )
        result[name] = target
    return result


def require_arm64_elf(data: bytes, label: str) -> None:
    """Check ELF64, little endian, AArch64 before building or publishing."""
    if len(data) < 20 or data[:6] != b"\x7fELF\x02\x01":
        raise RuntimeError(f"{label} is not a little-endian ELF64 binary")
    machine = struct.unpack_from("<H", data, 18)[0]
    if machine != 183:
        raise RuntimeError(f"{label} has ELF machine {machine}; expected AArch64 (183)")


def _prepare_source(archive: Path, module: Path, destination: Path) -> Path:
    """Discard the sdist builder's binary before assembling native build input."""
    with tarfile.open(archive) as source:
        source.extractall(destination, filter="data")
    root = destination / "falkordblite-0.10.0"
    (root / "falkordb.so").unlink(missing_ok=True)
    shutil.rmtree(root / "redislite" / "bin", ignore_errors=True)
    for metadata in root.glob("*.egg-info"):
        shutil.rmtree(metadata)
    (root / "PKG-INFO").unlink(missing_ok=True)
    require_arm64_elf(module.read_bytes(), MODULE)
    shutil.copyfile(module, root / "falkordb.so")

    config = configparser.ConfigParser()
    config.read(root / "setup.cfg")
    project = _project()
    config["metadata"]["name"] = project["name"]
    config["metadata"]["version"] = project["version"]
    config["metadata"]["description"] = project["description"]
    config["options"]["python_requires"] = project["requires-python"]
    config["options"]["install_requires"] = "\n" + "\n".join(project["dependencies"])
    with (root / "setup.cfg").open("w") as stream:
        config.write(stream)

    # Our published sdist is the recipe below, never this contaminated upstream
    # archive. Also remove inherited binary includes for any local source build.
    manifest = (root / "MANIFEST.in").read_text()
    manifest = manifest.replace("include falkordb.so", "exclude falkordb.so")
    manifest = manifest.replace(
        "recursive-include redislite/bin *", "prune redislite/bin"
    )
    (root / "MANIFEST.in").write_text(manifest)
    return root


def _verify_wheel(path: Path) -> None:
    with zipfile.ZipFile(path) as wheel:
        for name in ("redislite/bin/redis-server", "redislite/bin/falkordb.so"):
            require_arm64_elf(wheel.read(name), name)


def _write_metadata(wheel_path: Path, metadata_directory=None) -> None:
    """Make the built metadata identical to the portable PEP517 metadata hook."""
    # wheel unpack/pack updates every RECORD digest, including METADATA.
    with tempfile.TemporaryDirectory() as directory:
        subprocess.run(  # noqa: S603 - fixed interpreter/tools and verified build input
            [sys.executable, "-m", "wheel", "unpack", str(wheel_path), "-d", directory],
            check=True,
        )
        unpacked = Path(directory) / f"{DIST}-{VERSION}"
        if metadata_directory is None:
            prepare_metadata_for_build_wheel(unpacked)
        else:
            # PEP 517 requires preserving prepared metadata, including any
            # additional files, when a frontend supplies its dist-info path.
            shutil.copytree(
                metadata_directory, unpacked / DIST_INFO, dirs_exist_ok=True
            )
        provenance = {
            "sources": {
                name: {"url": url, "sha256": sha}
                for name, (url, sha) in SOURCES.items()
            },
            "repair": "Discard upstream sdist's x86-64 module; build Redis and package AArch64 module.",
        }
        (unpacked / "redislite" / "potpie_build.json").write_text(
            json.dumps(provenance, indent=2) + "\n"
        )
        subprocess.run(  # noqa: S603 - fixed interpreter/tools and verified build input
            [
                sys.executable,
                "-m",
                "wheel",
                "pack",
                str(unpacked),
                "-d",
                str(wheel_path.parent),
            ],
            check=True,
        )


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    if sys.platform != "linux" or platform.machine().lower() not in {
        "aarch64",
        "arm64",
    }:
        raise RuntimeError(
            "potpie-falkordblite wheels must be built on native Linux ARM64"
        )
    output = Path(wheel_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="potpie-falkordblite-") as temporary:
        work = Path(temporary)
        cache = Path(os.environ.get("POTPIE_NATIVE_SOURCE_CACHE", str(work / "inputs")))
        inputs = fetch_sources(cache)
        source = _prepare_source(inputs[SDIST], inputs[MODULE], work / "source")
        raw = work / "raw"
        raw.mkdir()
        # The selected interpreter's PEP517 build environment already contains
        # the pinned tools; invoke upstream setuptools only after input checks.
        subprocess.run(  # noqa: S603 - fixed interpreter/tools and verified build input
            [sys.executable, "setup.py", "bdist_wheel", "--dist-dir", str(raw)],
            cwd=source,
            check=True,
        )
        wheels = list(raw.glob("*.whl"))
        if len(wheels) != 1:
            raise RuntimeError(f"expected one raw native wheel, found {len(wheels)}")
        _verify_wheel(wheels[0])
        repaired = work / "repaired"
        # auditwheel validates symbol versions and bundles allowed non-system
        # libraries. It must fail rather than relabel an incompatible binary.
        subprocess.run(  # noqa: S603 - fixed interpreter/tools and verified build input
            [
                sys.executable,
                "-m",
                "auditwheel",
                "repair",
                "--plat",
                WHEEL_PLATFORM,
                # Keep the final tag identical to portable prepared metadata.
                "--only-plat",
                "--wheel-dir",
                str(repaired),
                str(wheels[0]),
            ],
            check=True,
        )
        wheels = list(repaired.glob("*.whl"))
        if len(wheels) != 1:
            raise RuntimeError(
                f"expected one repaired native wheel, found {len(wheels)}"
            )
        _verify_wheel(wheels[0])
        _write_metadata(wheels[0], metadata_directory)
        destination = output / wheels[0].name
        shutil.copyfile(wheels[0], destination)
    return destination.name


def build_sdist(sdist_directory, config_settings=None):
    """Publish only the build recipe; platform binaries cannot leak into it."""
    output = Path(sdist_directory)
    output.mkdir(parents=True, exist_ok=True)
    name = f"{DIST}-{VERSION}.tar.gz"
    prefix = f"{DIST}-{VERSION}"
    with tarfile.open(output / name, "w:gz") as archive:
        for filename in ("pyproject.toml", "_build.py", "README.md"):
            archive.add(ROOT / filename, arcname=f"{prefix}/{filename}")
        metadata = _metadata()
        info = tarfile.TarInfo(f"{prefix}/PKG-INFO")
        info.size = len(metadata)
        archive.addfile(info, io.BytesIO(metadata))
    return name
