# Running gVisor on Mac

## Overview

gVisor (`runsc`) is a Linux-specific technology and **does not run natively on macOS**. However, you can use gVisor on Mac through **Docker Desktop**, which runs a Linux virtual machine.

## Option 1: Docker Desktop with gVisor Runtime (Recommended)

Docker Desktop on Mac runs a Linux VM, so you can configure gVisor to work inside that VM.

### Setup Steps

1. **Install Docker Desktop** (if not already installed)

   ```bash
   # Download from: https://www.docker.com/products/docker-desktop
   # Or install via Homebrew:
   brew install --cask docker
   ```

2. **Install gVisor in Docker Desktop's Linux VM**

   You need to install `runsc` inside the Docker Desktop VM. This is a bit complex:

   ```bash
   # Option A: Use a helper script (if available)
   # Option B: SSH into Docker Desktop VM and install manually
   # Option C: Use a Docker container to install runsc
   ```

   **Simplest approach**: Install runsc in a container and configure Docker to use it:

   ```bash
   # Download runsc for Linux
   ARCH="x86_64"  # or "arm64" for Apple Silicon
   URL=https://storage.googleapis.com/gvisor/releases/release/latest/${ARCH}
   wget ${URL}/runsc ${URL}/runsc.sha512
   sha512sum -c runsc.sha512
   chmod a+rx runsc

   # Copy runsc into Docker Desktop VM
   # This requires accessing Docker Desktop's VM filesystem
   # On Mac, Docker Desktop stores files in:
   # ~/Library/Containers/com.docker.docker/Data/vms/0/
   ```

3. **Configure Docker to use runsc runtime**

   Edit Docker Desktop settings or create/edit `~/.docker/daemon.json`:

   ```json
   {
     "runtimes": {
       "runsc": {
         "path": "/usr/local/bin/runsc"
       }
     }
   }
   ```

   Then restart Docker Desktop.

4. **Verify setup**

   ```bash
   docker info --format "{{.Runtimes}}"
   # Should show "runsc" in the output

   # Test with a container
   docker run --rm --runtime=runsc busybox echo "Hello from gVisor"
   ```

### Limitations

- **Complex setup**: Requires manual installation in Docker Desktop's VM
- **Performance**: Slight overhead from VM + gVisor
- **Maintenance**: Need to update runsc manually when Docker Desktop updates

## Option 2: Use a Linux VM

Run a Linux VM on your Mac (using VirtualBox, VMware, Parallels, etc.) and install gVisor there.

### Setup Steps

1. **Install a Linux VM** (Ubuntu recommended)
2. **Install gVisor in the VM** following standard Linux instructions
3. **Use the VM** for development/testing

## Option 3: Use Remote Linux Machine

Develop on a remote Linux machine (cloud instance, remote server, etc.) where gVisor runs natively.

## Option 4: Use the Fallback (When Docker Desktop + runsc is Not Configured)

If Docker Desktop is not configured with the runsc runtime, the system **automatically falls back to regular subprocess execution on Mac**, which is:

- ✅ **Simple**: No setup required
- ✅ **Works immediately**: Commands execute normally
- ✅ **Secure enough for local dev**: Regular subprocess is fine for local development
- ✅ **Same API**: Your code works the same way

**Note**: gVisor is fully supported on Mac when Docker Desktop is configured with runsc (see Option 1). The fallback only occurs when Docker Desktop is not available or runsc is not configured.

### When to Use Each Option

| Option                                 | Best For                                                             | Complexity                 |
| -------------------------------------- | -------------------------------------------------------------------- | -------------------------- |
| **Docker Desktop + gVisor**            | Full gVisor support on Mac (recommended for production-like testing) | Medium (with setup script) |
| **Linux VM**                           | Full Linux development environment                                   | Medium                     |
| **Remote Linux**                       | Production-like testing                                              | Low (if you have access)   |
| **Fallback (no Docker Desktop/runsc)** | Local Mac development without gVisor                                 | None                       |

## Recommendation

For **local Mac development**:

- **With gVisor**: Use Docker Desktop + runsc runtime (Option 1) - fully supported and recommended for testing gVisor behavior
- **Without gVisor**: The automatic fallback works seamlessly - no setup required
- **In K8s (Linux)**: gVisor will be used automatically

If you need to **test gVisor behavior specifically**, use:

- Docker Desktop + runsc (Option 1) - fully supported on Mac
- A Linux VM, or
- A remote Linux machine, or
- Test in your K8s environment where gVisor is already configured

## Testing on Mac

The current implementation will:

1. Detect Mac platform
2. Check for Docker Desktop with runsc runtime
3. If Docker Desktop + runsc is configured: Use gVisor (fully supported)
4. If not available: Use regular subprocess (automatic fallback)

You can verify this works:

```python
from app.modules.utils.gvisor_runner import is_gvisor_available, run_command_isolated

# Check availability
print(f"gVisor available: {is_gvisor_available()}")  # False on Mac (unless Docker+runsc configured)

# Commands still work
result = run_command_isolated(["echo", "Hello"])
print(result.stdout)  # Works fine with fallback
```

## Summary

- **Native gVisor on Mac**: ❌ Not possible (Linux-only)
- **gVisor via Docker Desktop**: ✅ Fully supported - use `bash scripts/setup_gvisor_docker.sh` to configure
- **Fallback (no Docker Desktop/runsc)**: ✅ Works seamlessly - no setup required
- **K8s deployment**: ✅ gVisor works automatically (Linux containers)

**gVisor is fully supported on Mac via Docker Desktop**. The setup script makes configuration straightforward. If you prefer not to configure Docker Desktop with runsc, the system automatically falls back to regular subprocess execution.
