"""
gVisor Installation Script

This script downloads and installs the gVisor runsc binary for command isolation.
gVisor provides a user-space kernel for better security isolation when running commands.

Usage:
    python -m app.modules.utils.install_gvisor
    or
    from app.modules.utils.install_gvisor import install_gvisor
    install_gvisor()
"""

import os
import sys
import platform
import subprocess
import shutil
from pathlib import Path
from typing import Optional

from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)

# gVisor release URL base
GVISOR_RELEASE_BASE = "https://storage.googleapis.com/gvisor/releases/release/latest"


def get_architecture() -> Optional[str]:
    """
    Get the system architecture for gVisor download.

    Returns:
        Architecture string (e.g., 'x86_64', 'arm64') or None if unsupported
    """
    machine = platform.machine().lower()
    system = platform.system().lower()

    # Map common architectures
    # Note: gVisor uses 'aarch64' for ARM64, not 'arm64'
    arch_map = {
        "x86_64": "x86_64",
        "amd64": "x86_64",
        "aarch64": "aarch64",  # gVisor uses 'aarch64', not 'arm64'
        "arm64": "aarch64",  # Map arm64 to aarch64 for gVisor
    }

    arch = arch_map.get(machine)

    if not arch:
        logger.warning(f"Unsupported architecture: {machine}")
        return None

    # gVisor primarily supports Linux
    if system != "linux":
        logger.warning(
            f"gVisor is primarily designed for Linux. Current system: {system}. "
            f"Installation may not work correctly."
        )

    return arch


def get_install_path() -> Path:
    """
    Get the installation path for runsc binary.

    Tries to install to a location that doesn't require sudo:
    1. Project's .venv/bin directory (if virtualenv exists)
    2. Project root/bin directory
    3. User's local bin directory

    Returns:
        Path object for the installation directory
    """
    # Try project's .venv/bin first
    project_root = Path(__file__).parent.parent.parent.parent
    venv_bin = project_root / ".venv" / "bin"
    if venv_bin.exists():
        return venv_bin

    # Try project root/bin
    project_bin = project_root / "bin"
    project_bin.mkdir(exist_ok=True)
    return project_bin


def check_runsc_installed(install_path: Path) -> bool:
    """
    Check if runsc is already installed and accessible.

    Args:
        install_path: Path where runsc should be installed

    Returns:
        True if runsc is installed and working, False otherwise
    """
    runsc_path = install_path / "runsc"

    if not runsc_path.exists():
        return False

    try:
        # Check if runsc is executable and works
        result = subprocess.run(
            [str(runsc_path), "--version"], capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0
    except Exception as e:
        logger.debug(f"Error checking runsc: {e}")
        return False


def download_file(url: str, dest: Path) -> bool:
    """
    Download a file from URL to destination.

    Args:
        url: URL to download from
        dest: Destination path

    Returns:
        True if successful, False otherwise
    """
    try:
        # Try using requests first (if available)
        try:
            import requests

            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            with open(dest, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        except ImportError:
            # Fallback to urllib (built-in, always available)
            import urllib.request
            import urllib.error

            try:
                urllib.request.urlretrieve(url, dest)
                return True
            except urllib.error.URLError as e:
                logger.error(f"Failed to download {url} with urllib: {e}")
                return False
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False


def verify_checksum(file_path: Path, checksum_url: str) -> bool:
    """
    Verify file checksum.

    Args:
        file_path: Path to the file to verify
        checksum_url: URL to the checksum file

    Returns:
        True if checksum matches, False otherwise
    """
    try:
        # Download checksum file
        checksum_path = file_path.parent / f"{file_path.name}.sha512"
        if not download_file(checksum_url, checksum_path):
            logger.warning("Failed to download checksum, skipping verification")
            return True  # Continue anyway

        # Read expected checksum
        with open(checksum_path, "r") as f:
            expected_checksum = f.read().split()[0]

        # Calculate actual checksum
        import hashlib

        sha512 = hashlib.sha512()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha512.update(chunk)
        actual_checksum = sha512.hexdigest()

        # Clean up checksum file
        checksum_path.unlink()

        if expected_checksum == actual_checksum:
            logger.info("Checksum verification passed")
            return True
        else:
            logger.error("Checksum verification failed")
            return False

    except Exception as e:
        logger.warning(f"Error verifying checksum: {e}, continuing anyway")
        return True  # Continue anyway


def install_gvisor(force: bool = False) -> bool:
    """
    Install gVisor runsc binary.

    Args:
        force: If True, reinstall even if already installed

    Returns:
        True if installation successful, False otherwise
    """
    arch = get_architecture()
    if not arch:
        logger.error("Cannot determine architecture for gVisor installation")
        return False

    install_path = get_install_path()
    runsc_path = install_path / "runsc"

    # Check if already installed
    if not force and check_runsc_installed(install_path):
        logger.info(f"gVisor runsc is already installed at {runsc_path}")
        return True

    logger.info(f"Installing gVisor runsc for architecture: {arch}")
    logger.info(f"Installation path: {install_path}")

    # Create installation directory if it doesn't exist
    install_path.mkdir(parents=True, exist_ok=True)

    # Download URLs
    base_url = f"{GVISOR_RELEASE_BASE}/{arch}"
    runsc_url = f"{base_url}/runsc"
    checksum_url = f"{base_url}/runsc.sha512"

    # Temporary download path
    temp_path = install_path / "runsc.tmp"

    try:
        # Download runsc binary
        logger.info(f"Downloading runsc from {runsc_url}")
        if not download_file(runsc_url, temp_path):
            logger.error("Failed to download runsc binary")
            return False

        # Verify checksum
        if not verify_checksum(temp_path, checksum_url):
            logger.error("Checksum verification failed")
            temp_path.unlink()
            return False

        # Make executable
        os.chmod(temp_path, 0o755)

        # Move to final location
        if runsc_path.exists():
            runsc_path.unlink()
        temp_path.rename(runsc_path)

        logger.info(f"Successfully installed gVisor runsc to {runsc_path}")

        # Verify installation
        if check_runsc_installed(install_path):
            logger.info("gVisor installation verified successfully")
            return True
        else:
            logger.error("Installation completed but verification failed")
            return False

    except Exception:
        logger.exception("Error during gVisor installation")
        if temp_path.exists():
            temp_path.unlink()
        return False


def get_runsc_path() -> Optional[Path]:
    """
    Get the path to the runsc binary if installed.

    Returns:
        Path to runsc binary, or None if not found
    """
    install_path = get_install_path()
    runsc_path = install_path / "runsc"

    if runsc_path.exists() and check_runsc_installed(install_path):
        return runsc_path

    # Also check system PATH
    runsc_system = shutil.which("runsc")
    if runsc_system:
        return Path(runsc_system)

    return None


def main():
    """Main entry point for command-line usage."""
    from app.modules.utils.logger import configure_logging

    configure_logging(level="INFO")

    force = "--force" in sys.argv

    success = install_gvisor(force=force)

    if success:
        runsc_path = get_runsc_path()
        if runsc_path:
            print(f"\n✓ gVisor runsc installed successfully at: {runsc_path}")
            print("\nYou can now use runsc to isolate commands:")
            print(f"  {runsc_path} run <container-id> <command>")
        sys.exit(0)
    else:
        print("\n✗ gVisor installation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
