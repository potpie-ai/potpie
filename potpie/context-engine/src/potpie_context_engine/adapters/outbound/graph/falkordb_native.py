"""Diagnose incompatible Linux binaries before starting the embedded server.

FalkorDBLite can build Redis from source while reusing a bundled module from
another architecture. Comparing the two ELF headers catches that packaging
error without loading native code or guessing which CPU emulation is available.
Other loader failures (including missing libraries) remain the driver's errors.
"""

from __future__ import annotations

import platform
import shutil
from importlib import metadata
from pathlib import Path

from potpie_context_engine.core.errors import GraphSubstrateUnavailable


def validate_falkordb_lite_installation() -> None:
    """Reject two distributions owning ``redislite`` before importing either.

    Pip treats the upstream and corrected distributions as different projects,
    although their files overlap. A later uninstall could remove the active
    driver's files, so require an explicit clean replacement of the upstream
    distribution instead of accepting whichever files were installed last.
    """
    if platform.system() != "Linux":
        return
    try:
        upstream = metadata.distribution("falkordblite")
        corrected = metadata.distribution("potpie-falkordblite")
    except metadata.PackageNotFoundError:
        return

    corrected_pin = f"potpie-falkordblite=={corrected.version}"
    raise GraphSubstrateUnavailable(
        f"Both falkordblite=={upstream.version} and {corrected_pin} are "
        "installed and own the same redislite files. The graph store was "
        "not opened.",
        recommended_next_action=(
            "Using the Python environment that runs Potpie, run "
            "`python -m pip uninstall falkordblite`, then "
            f"`python -m pip install --force-reinstall --no-deps {corrected_pin}` "
            "with the same trusted index or --find-links wheel directory used "
            "for the corrected package. Complete the reinstall before starting "
            "Potpie again; keep the existing graph files."
        ),
    )


def _elf_architecture(path: str) -> tuple[int, int, int] | None:
    """Read ELF class, byte order and machine; leave other formats to the OS."""
    try:
        with Path(path).open("rb") as binary:
            header = binary.read(20)
    except OSError:
        return None
    if (
        len(header) < 20
        or header[:4] != b"\x7fELF"
        or header[4] not in (1, 2)
        or header[5] not in (1, 2)
    ):
        return None
    byteorder = "little" if header[5] == 1 else "big"
    return header[4], header[5], int.from_bytes(header[18:20], byteorder)


def _describe(architecture: tuple[int, int, int]) -> str:
    elf_class, byte_order, machine = architecture
    name = {3: "x86", 40: "ARM", 62: "x86-64", 183: "ARM64"}.get(
        machine, f"ELF machine {machine}"
    )
    endian = "little-endian" if byte_order == 1 else "big-endian"
    return f"{name}, {32 if elf_class == 1 else 64}-bit {endian}"


def validate_falkordb_lite_binaries() -> None:
    """Reject a proven Redis/module architecture mismatch on Linux only.

    Check the paths used by ``redislite.client`` itself. Matching foreign
    binaries may work under emulation, so their architecture need not equal the
    host CPU. Missing or unrecognised binaries retain existing driver errors.
    """
    if platform.system() != "Linux":
        return

    from redislite import client

    executable = client.__redis_executable__ or shutil.which("redis-server")
    module = client.__falkordb_module__
    if not executable or not module:
        return
    server_arch = _elf_architecture(executable)
    module_arch = _elf_architecture(module)
    if server_arch is None or module_arch is None or server_arch == module_arch:
        return

    raise GraphSubstrateUnavailable(
        "FalkorDBLite has incompatible native binaries: "
        f"Redis at {executable!r} is {_describe(server_arch)}, but "
        f"the graph module at {module!r} is {_describe(module_arch)}. "
        "The graph store was not opened.",
        recommended_next_action=(
            "Install a corrected FalkorDBLite wheel containing Redis and the "
            "graph module for the same Linux architecture. An ordinary "
            "reinstall of the affected source package can reproduce this "
            "packaging error; keep the existing graph files."
        ),
    )
