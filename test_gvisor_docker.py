#!/usr/bin/env python3
"""
Test gVisor through Docker on Mac
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.modules.utils.gvisor_runner import (
    is_gvisor_available,
    run_command_isolated,
    run_shell_command_isolated,
    _check_docker_available,
    _is_docker_desktop,
)


def test_gvisor_setup():
    """Test gVisor setup and availability."""
    print("=" * 60)
    print("gVisor Docker Setup Test")
    print("=" * 60)
    print()

    print("1. Checking Docker availability...")
    docker_available = _check_docker_available()
    print(f"   Docker with runsc runtime: {docker_available}")
    print()

    print("2. Checking if Docker Desktop...")
    is_desktop = _is_docker_desktop()
    print(f"   Docker Desktop detected: {is_desktop}")
    print()

    print("3. Checking gVisor availability...")
    gvisor_available = is_gvisor_available()
    print(f"   gVisor available: {gvisor_available}")
    print()

    return gvisor_available


def test_commands():
    """Test running commands through gVisor."""
    print("=" * 60)
    print("Testing Commands Through gVisor")
    print("=" * 60)
    print()

    # Test 1: Simple echo command
    print("Test 1: Simple echo command")
    result = run_command_isolated(
        command=["echo", "Hello from gVisor test"],
        use_gvisor=True,
    )
    print(f"   Return code: {result.returncode}")
    print(f"   Success: {result.success}")
    print(f"   Stdout: {result.stdout.strip()}")
    if result.stderr:
        print(f"   Stderr: {result.stderr.strip()}")
    assert result.success, "Command should succeed"
    assert "Hello from gVisor test" in result.stdout
    print("   ✓ Passed")
    print()

    # Test 2: Shell command with multiple commands
    print("Test 2: Shell command (date + echo)")
    result = run_shell_command_isolated(
        shell_command="date && echo 'Command executed'",
        use_gvisor=True,
    )
    print(f"   Return code: {result.returncode}")
    print(f"   Success: {result.success}")
    print(f"   Stdout: {result.stdout.strip()}")
    assert result.success, "Shell command should succeed"
    assert "Command executed" in result.stdout
    print("   ✓ Passed")
    print()

    # Test 3: Command with working directory
    import tempfile
    import os

    print("Test 3: Command with working directory")
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "test.txt")
        with open(test_file, "w") as f:
            f.write("test content")

        result = run_command_isolated(
            command=["cat", "test.txt"],
            working_dir=tmpdir,
            use_gvisor=True,
        )
        print(f"   Return code: {result.returncode}")
        print(f"   Success: {result.success}")
        print(f"   Stdout: {result.stdout.strip()}")
        assert result.success, "Command with working dir should succeed"
        assert "test content" in result.stdout
        print("   ✓ Passed")
    print()

    # Test 4: List files
    print("Test 4: List files (ls)")
    result = run_command_isolated(
        command=["ls", "-la", "/"],
        use_gvisor=True,
    )
    print(f"   Return code: {result.returncode}")
    print(f"   Success: {result.success}")
    print(f"   Stdout (first 200 chars): {result.stdout[:200]}...")
    assert result.success, "ls command should succeed"
    print("   ✓ Passed")
    print()

    # Test 5: Environment variables
    print("Test 5: Environment variables")
    result = run_command_isolated(
        command=["sh", "-c", "echo $TEST_VAR"],
        env={"TEST_VAR": "test_value"},
        use_gvisor=True,
    )
    print(f"   Return code: {result.returncode}")
    print(f"   Success: {result.success}")
    print(f"   Stdout: {result.stdout.strip()}")
    assert result.success, "Command with env var should succeed"
    assert "test_value" in result.stdout
    print("   ✓ Passed")
    print()


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("gVisor Docker Test Suite")
    print("=" * 60)
    print()

    try:
        # Test setup
        gvisor_available = test_gvisor_setup()

        if not gvisor_available:
            print("⚠️  gVisor is not available.")
            print()
            print("To enable gVisor on Mac:")
            print("1. Open Docker Desktop Settings > Docker Engine")
            print("2. Add this to the JSON:")
            print("   {")
            print('     "runtimes": {')
            print('       "runsc": {')
            print('         "path": "/usr/local/bin/runsc"')
            print("       }")
            print("     }")
            print("   }")
            print("3. Click 'Apply & Restart'")
            print("4. Run this test again")
            print()
            return 1

        # Test commands
        test_commands()

        print("=" * 60)
        print("All Tests Passed! ✓")
        print("=" * 60)
        print()
        print("gVisor is working correctly through Docker on Mac!")
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
