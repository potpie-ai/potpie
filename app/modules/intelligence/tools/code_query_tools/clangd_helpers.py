"""
Helper functions for clangd setup and compile_commands.json generation.

This module provides utilities to automatically generate compile_commands.json
for C/C++ projects using various build systems and tools.
"""

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def find_compile_commands(workspace_root: str) -> Optional[Path]:
    """
    Find compile_commands.json in common locations.
    
    Returns:
        Path to compile_commands.json if found, None otherwise
    """
    workspace_path = Path(workspace_root)
    
    # Check common locations
    locations = [
        workspace_path / "compile_commands.json",
        workspace_path / "build" / "compile_commands.json",
        workspace_path / "cmake-build-debug" / "compile_commands.json",
        workspace_path / "cmake-build-release" / "compile_commands.json",
    ]
    
    for location in locations:
        if location.exists() and location.is_file():
            logger.info(f"[clangd] Found compile_commands.json at {location}")
            return location
    
    return None


def detect_build_system(workspace_root: str) -> Optional[str]:
    """
    Detect the build system used by the project.
    
    Returns:
        'cmake', 'make', 'autotools', or None
    """
    workspace_path = Path(workspace_root)
    
    # Check for CMake
    if (workspace_path / "CMakeLists.txt").exists():
        logger.info("[clangd] Detected CMake build system")
        return "cmake"
    
    # Check for Makefile
    if (workspace_path / "Makefile").exists() or (workspace_path / "makefile").exists():
        logger.info("[clangd] Detected Make build system")
        return "make"
    
    # Check for autotools
    if (workspace_path / "configure.ac").exists() or (workspace_path / "configure.in").exists():
        logger.info("[clangd] Detected autotools build system")
        return "autotools"
    
    return None


def generate_compile_commands_cmake(workspace_root: str) -> Optional[Path]:
    """
    Generate compile_commands.json for CMake projects.
    
    Returns:
        Path to generated compile_commands.json, or None if failed
    """
    workspace_path = Path(workspace_root)
    cmake_lists = workspace_path / "CMakeLists.txt"
    
    if not cmake_lists.exists():
        return None
    
    # Check if cmake is available
    cmake_path = shutil.which("cmake")
    if not cmake_path:
        logger.warning("[clangd] CMake not found in PATH, cannot generate compile_commands.json")
        return None
    
    # Try to find existing build directory or create one
    build_dir = workspace_path / "build"
    if not build_dir.exists():
        build_dir = workspace_path / "cmake-build-debug"
    
    # Create build directory if it doesn't exist
    if not build_dir.exists():
        try:
            build_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(f"[clangd] Failed to create build directory: {e}")
            return None
    
    try:
        # Run cmake with compile commands export
        logger.info(f"[clangd] Running CMake to generate compile_commands.json in {build_dir}")
        result = subprocess.run(
            [cmake_path, "-DCMAKE_EXPORT_COMPILE_COMMANDS=1", ".."],
            cwd=build_dir,
            capture_output=True,
            text=True,
            timeout=60,
        )
        
        if result.returncode != 0:
            logger.warning(
                f"[clangd] CMake failed: {result.stderr[:500]}"
            )
            return None
        
        # Check if compile_commands.json was created
        compile_commands = build_dir / "compile_commands.json"
        if compile_commands.exists():
            logger.info(f"[clangd] Successfully generated compile_commands.json at {compile_commands}")
            
            # Symlink to workspace root for easier access
            root_compile_commands = workspace_path / "compile_commands.json"
            if not root_compile_commands.exists():
                try:
                    root_compile_commands.symlink_to(compile_commands.relative_to(workspace_path))
                    logger.info(f"[clangd] Created symlink: {root_compile_commands} -> {compile_commands}")
                except Exception as e:
                    logger.debug(f"[clangd] Could not create symlink: {e}")
            
            return compile_commands
        
    except subprocess.TimeoutExpired:
        logger.warning("[clangd] CMake command timed out")
    except Exception as e:
        logger.warning(f"[clangd] Failed to run CMake: {e}")
    
    return None


def generate_compile_commands_bear(workspace_root: str) -> Optional[Path]:
    """
    Generate compile_commands.json for Make-based projects using Bear.
    
    Returns:
        Path to generated compile_commands.json, or None if failed
    """
    workspace_path = Path(workspace_root)
    
    # Check if bear is available
    bear_path = shutil.which("bear")
    if not bear_path:
        logger.warning("[clangd] Bear not found in PATH, cannot generate compile_commands.json")
        logger.info("[clangd] Install Bear with: brew install bear (macOS) or apt install bear (Linux)")
        return None
    
    # Check for Makefile
    makefile = workspace_path / "Makefile"
    if not makefile.exists():
        makefile = workspace_path / "makefile"
    
    if not makefile.exists():
        return None
    
    try:
        # Run bear with make
        logger.info(f"[clangd] Running Bear to generate compile_commands.json")
        result = subprocess.run(
            [bear_path, "--", "make", "clean"],
            cwd=workspace_path,
            capture_output=True,
            text=True,
            timeout=120,
        )
        
        # Clean might fail, that's okay - try building
        result = subprocess.run(
            [bear_path, "--", "make", "-j4"],
            cwd=workspace_path,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutes for build
        )
        
        # Check if compile_commands.json was created
        compile_commands = workspace_path / "compile_commands.json"
        if compile_commands.exists():
            logger.info(f"[clangd] Successfully generated compile_commands.json using Bear")
            return compile_commands
        
        if result.returncode != 0:
            logger.warning(
                f"[clangd] Bear build failed: {result.stderr[:500]}"
            )
        
    except subprocess.TimeoutExpired:
        logger.warning("[clangd] Bear command timed out")
    except Exception as e:
        logger.warning(f"[clangd] Failed to run Bear: {e}")
    
    return None


def generate_compile_commands_compiledb(workspace_root: str) -> Optional[Path]:
    """
    Generate compile_commands.json using compiledb (Python tool).
    
    Returns:
        Path to generated compile_commands.json, or None if failed
    """
    workspace_path = Path(workspace_root)
    
    # Check if compiledb is available
    compiledb_path = shutil.which("compiledb")
    if not compiledb_path:
        logger.warning("[clangd] compiledb not found in PATH")
        return None
    
    # Check for Makefile
    makefile = workspace_path / "Makefile"
    if not makefile.exists():
        makefile = workspace_path / "makefile"
    
    if not makefile.exists():
        return None
    
    try:
        # Run compiledb with make
        logger.info(f"[clangd] Running compiledb to generate compile_commands.json")
        result = subprocess.run(
            [compiledb_path, "-n", "make", "-j4"],
            cwd=workspace_path,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutes for build
        )
        
        # Check if compile_commands.json was created
        compile_commands = workspace_path / "compile_commands.json"
        if compile_commands.exists():
            logger.info(f"[clangd] Successfully generated compile_commands.json using compiledb")
            return compile_commands
        
        if result.returncode != 0:
            logger.warning(
                f"[clangd] compiledb build failed: {result.stderr[:500]}"
            )
        
    except subprocess.TimeoutExpired:
        logger.warning("[clangd] compiledb command timed out")
    except Exception as e:
        logger.warning(f"[clangd] Failed to run compiledb: {e}")
    
    return None


def create_compile_flags_txt(workspace_root: str, language: str = "c") -> Optional[Path]:
    """
    Create a basic compile_flags.txt file as a fallback.
    
    This is a simple alternative to compile_commands.json that works
    for projects with consistent compiler flags.
    
    Returns:
        Path to created compile_flags.txt, or None if failed
    """
    workspace_path = Path(workspace_root)
    compile_flags = workspace_path / "compile_flags.txt"
    
    # Don't overwrite existing file
    if compile_flags.exists():
        logger.info(f"[clangd] compile_flags.txt already exists at {compile_flags}")
        return compile_flags
    
    # Detect include directories
    include_dirs = []
    for include_dir in ["include", "inc", "src", "."]:
        include_path = workspace_path / include_dir
        if include_path.exists() and include_path.is_dir():
            include_dirs.append(f"-I{include_dir}")
    
    # Set C standard based on language
    std_flag = "-std=c11" if language == "c" else "-std=c++17"
    
    # Create basic compile_flags.txt
    flags = [
        f"-x{language}",  # Language type
        std_flag,
        "-Wall",
    ] + include_dirs
    
    try:
        with open(compile_flags, "w") as f:
            f.write("\n".join(flags) + "\n")
        
        logger.info(f"[clangd] Created compile_flags.txt at {compile_flags}")
        return compile_flags
    except Exception as e:
        logger.warning(f"[clangd] Failed to create compile_flags.txt: {e}")
        return None


def ensure_compile_commands(
    workspace_root: str, language: str = "c", force_regenerate: bool = False
) -> Tuple[Optional[Path], list]:
    """
    Ensure compile_commands.json exists, generating it if necessary.
    
    This function tries multiple strategies:
    1. Find existing compile_commands.json
    2. Detect build system and generate using appropriate tool
    3. Create compile_flags.txt as fallback
    
    Args:
        workspace_root: Root directory of the workspace
        language: Language type ('c' or 'cpp')
        force_regenerate: If True, regenerate even if file exists
    
    Returns:
        Tuple of (Path to compile_commands.json or compile_flags.txt, status_messages)
    """
    status_messages = []
    workspace_path = Path(workspace_root)
    
    # Step 1: Check if compile_commands.json already exists
    if not force_regenerate:
        existing = find_compile_commands(workspace_root)
        if existing:
            status_messages.append(f"Found existing compile_commands.json at {existing}")
            return existing, status_messages
    
    # Step 2: Detect build system and generate
    build_system = detect_build_system(workspace_root)
    
    if build_system == "cmake":
        status_messages.append("Detected CMake project, attempting to generate compile_commands.json...")
        result = generate_compile_commands_cmake(workspace_root)
        if result:
            status_messages.append(f"Successfully generated compile_commands.json using CMake")
            return result, status_messages
        status_messages.append("CMake generation failed, trying other methods...")
    
    elif build_system == "make":
        status_messages.append("Detected Make project, attempting to generate compile_commands.json...")
        
        # Try Bear first (most reliable)
        result = generate_compile_commands_bear(workspace_root)
        if result:
            status_messages.append(f"Successfully generated compile_commands.json using Bear")
            return result, status_messages
        
        # Try compiledb as fallback
        result = generate_compile_commands_compiledb(workspace_root)
        if result:
            status_messages.append(f"Successfully generated compile_commands.json using compiledb")
            return result, status_messages
        
        status_messages.append("Make-based generation failed (Bear/compiledb not available or build failed)")
    
    # Step 3: Create compile_flags.txt as fallback
    status_messages.append("Creating compile_flags.txt as fallback...")
    result = create_compile_flags_txt(workspace_root, language)
    if result:
        status_messages.append(f"Created compile_flags.txt (simpler alternative to compile_commands.json)")
        return result, status_messages
    
    status_messages.append("Warning: Could not generate compile_commands.json or compile_flags.txt")
    return None, status_messages

