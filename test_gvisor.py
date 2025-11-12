#!/usr/bin/env python3
"""
Test script for gVisor functionality.
Tests that gVisor detection and fallback work correctly on Mac/Windows.
"""

import sys
import platform
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.modules.utils.gvisor_runner import (
    is_gvisor_available,
    run_command_isolated,
    run_shell_command_isolated,
    get_runsc_binary,
    _is_running_in_container,
)


def test_platform_detection():
    """Test that platform detection works correctly."""
    print("=" * 60)
    print("Platform Detection Test")
    print("=" * 60)
    print(f"Platform: {platform.system()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Running in container: {_is_running_in_container()}")
    print()


def test_gvisor_availability():
    """Test gVisor availability detection."""
    print("=" * 60)
    print("gVisor Availability Test")
    print("=" * 60)

    available = is_gvisor_available()
    runsc_path = get_runsc_binary()

    print(f"gVisor available: {available}")
    print(f"runsc binary path: {runsc_path}")

    if platform.system().lower() != "linux":
        print(f"✓ Expected: gVisor not available on {platform.system()}")
        assert not available, "gVisor should not be available on non-Linux platforms"
    else:
        print("Platform is Linux - gVisor may be available if installed")

    print()


def test_command_execution():
    """Test that command execution works with fallback."""
    print("=" * 60)
    print("Command Execution Test")
    print("=" * 60)

    # Test 1: Simple command
    print("Test 1: Simple echo command")
    result = run_command_isolated(
        command=["echo", "Hello from gVisor test"],
        use_gvisor=True,  # Try to use gVisor (will fall back on Mac)
    )
    print(f"  Return code: {result.returncode}")
    print(f"  Success: {result.success}")
    print(f"  Stdout: {result.stdout.strip()}")
    if result.stderr:
        print(f"  Stderr: {result.stderr.strip()}")
    assert result.success, "Command should succeed"
    assert "Hello from gVisor test" in result.stdout
    print("  ✓ Passed")
    print()

    # Test 2: Shell command
    print("Test 2: Shell command")
    result = run_shell_command_isolated(
        shell_command="echo 'Shell test' && echo 'Multiple lines'",
        use_gvisor=True,
    )
    print(f"  Return code: {result.returncode}")
    print(f"  Success: {result.success}")
    print(f"  Stdout: {result.stdout.strip()}")
    assert result.success, "Shell command should succeed"
    print("  ✓ Passed")
    print()

    # Test 3: Command with working directory
    print("Test 3: Command with working directory")
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test.txt"
        test_file.write_text("test content")

        result = run_command_isolated(
            command=["cat", "test.txt"],
            working_dir=str(tmpdir),
            use_gvisor=True,
        )
        print(f"  Return code: {result.returncode}")
        print(f"  Success: {result.success}")
        print(f"  Stdout: {result.stdout.strip()}")
        assert result.success, "Command with working dir should succeed"
        assert "test content" in result.stdout
        print("  ✓ Passed")
    print()

    # Test 4: Force no gVisor
    print("Test 4: Force no gVisor (explicit fallback)")
    result = run_command_isolated(
        command=["echo", "No gVisor"],
        use_gvisor=False,  # Explicitly disable gVisor
    )
    print(f"  Return code: {result.returncode}")
    print(f"  Success: {result.success}")
    print(f"  Stdout: {result.stdout.strip()}")
    assert result.success, "Command without gVisor should succeed"
    assert "No gVisor" in result.stdout
    print("  ✓ Passed")
    print()


def test_error_handling():
    """Test error handling."""
    print("=" * 60)
    print("Error Handling Test")
    print("=" * 60)

    # Test: Non-existent command
    print("Test: Non-existent command")
    result = run_command_isolated(
        command=["nonexistent_command_xyz123"],
        use_gvisor=True,
    )
    print(f"  Return code: {result.returncode}")
    print(f"  Success: {result.success}")
    assert not result.success, "Non-existent command should fail"
    print("  ✓ Passed")
    print()

    # Test: Non-existent working directory
    print("Test: Non-existent working directory")
    result = run_command_isolated(
        command=["ls"],
        working_dir="/nonexistent/directory/xyz123",
        use_gvisor=True,
    )
    print(f"  Return code: {result.returncode}")
    print(f"  Success: {result.success}")
    assert not result.success, "Non-existent directory should fail"
    print("  ✓ Passed")
    print()


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("gVisor Test Suite - Mac/Windows Fallback Test")
    print("=" * 60)
    print()

    try:
        test_platform_detection()
        test_gvisor_availability()
        test_command_execution()
        test_error_handling()

        print("=" * 60)
        print("All Tests Passed! ✓")
        print("=" * 60)
        print()
        print("Summary:")
        print(f"  - Platform: {platform.system()}")
        print(f"  - gVisor available: {is_gvisor_available()}")
        print("  - Fallback working: ✓")
        print("  - Commands execute correctly: ✓")
        print()
        print("On Mac/Windows, gVisor is not available, but the system")
        print("correctly falls back to regular subprocess execution.")
        print()

        return 0

    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
