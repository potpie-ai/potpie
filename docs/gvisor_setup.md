# gVisor Setup Guide

This guide explains how to set up and use gVisor for command isolation in both Kubernetes (K8s) and local development environments.

## Overview

gVisor provides a user-space kernel for better security isolation when running commands. In this project, gVisor is used to isolate commands executed for repositories in `/.repos` through agents.

## Environment Support

- **K8s/Linux Containers**: gVisor is automatically installed in the Docker image and will be used when available
- **Local Linux**: Can use gVisor with Docker runtime (optional setup)
- **Local Mac/Windows**: gVisor is fully supported via Docker Desktop when configured with the runsc runtime. If Docker Desktop is not configured with runsc, the system automatically falls back to regular subprocess execution.

The system automatically detects the environment and uses the appropriate method.

## Installation

### K8s/Container Environments

gVisor is **automatically installed** in the Docker image during build. No additional setup is required in K8s - the container will have `runsc` available at `/usr/local/bin/runsc`.

The system will automatically:

- Detect that it's running in a container
- Use runsc directly for command isolation
- Fall back gracefully if runsc isn't available or doesn't work

### Local Development

#### Automatic Installation

The project includes an automatic installation script that runs during setup:

```bash
python scripts/install_gvisor.py
```

This script will:

- Detect your system architecture
- Download the latest `runsc` binary
- Install it to `.venv/bin/runsc` (or `bin/runsc` in project root)
- Verify the installation

### Manual Installation

#### Option 1: Using the Installation Script

```bash
# From project root
python scripts/install_gvisor.py
```

#### Option 2: Manual Installation via apt (Debian/Ubuntu)

```bash
# Add gVisor repository
sudo apt-get update && \
sudo apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg

curl -fsSL https://gvisor.dev/archive.key | sudo gpg --dearmor -o /usr/share/keyrings/gvisor-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/gvisor-archive-keyring.gpg] https://storage.googleapis.com/gvisor/releases release main" | sudo tee /etc/apt/sources.list.d/gvisor.list > /dev/null

sudo apt-get update && sudo apt-get install -y runsc
```

#### Option 3: Manual Binary Installation

```bash
# Download and install runsc binary
ARCH=$(uname -m)
URL=https://storage.googleapis.com/gvisor/releases/release/latest/${ARCH}
wget ${URL}/runsc ${URL}/runsc.sha512
sha512sum -c runsc.sha512
chmod a+rx runsc
sudo mv runsc /usr/local/bin
```

## Docker Integration (Recommended)

For command isolation, gVisor works best when integrated with Docker:

### 1. Install gVisor as Docker Runtime

```bash
# If runsc is in /usr/local/bin
sudo /usr/local/bin/runsc install

# Or if runsc is in project's .venv/bin
sudo .venv/bin/runsc install
```

### 2. Reload Docker Daemon

```bash
sudo systemctl reload docker
```

### 3. Verify Installation

```bash
# Check if runsc runtime is available
docker info --format "{{.Runtimes}}"

# Test with a simple container
docker run --rm --runtime=runsc busybox echo "Hello from gVisor"
```

### 4. Configure Docker (Optional)

If needed, you can explicitly configure Docker to recognize runsc:

Edit `/etc/docker/daemon.json`:

```json
{
  "runtimes": {
    "runsc": {
      "path": "/usr/local/bin/runsc"
    }
  }
}
```

Then restart Docker:

```bash
sudo systemctl restart docker
```

## Docker Desktop Integration (Mac/Windows)

For Mac/Windows users, gVisor can work through Docker Desktop, which runs a Linux VM:

### Setup Steps

1. **Install Docker Desktop** (if not already installed)

   - Download from: https://www.docker.com/products/docker-desktop
   - Or install via Homebrew: `brew install --cask docker`

2. **Install gVisor in Docker Desktop**

   Use the provided setup script:

   ```bash
   bash scripts/setup_gvisor_docker.sh
   ```

   Or follow the detailed guide: [docker_desktop_gvisor_config.md](./docker_desktop_gvisor_config.md)

3. **Restart Docker Desktop**

   After running the setup script, restart Docker Desktop completely for changes to take effect.

4. **Verify Installation**

   ```bash
   # Check if runsc runtime is available
   docker info --format "{{.Runtimes}}" | grep runsc

   # Test with a simple container
   docker run --rm --runtime=runsc busybox echo "Hello from gVisor"
   ```

**Note**: If Docker Desktop + runsc is not configured, the system will automatically fall back to regular subprocess execution, which works seamlessly for local development.

## Usage in Code

### Basic Usage

```python
from app.modules.utils.gvisor_runner import run_command_isolated

# Run a command in an isolated gVisor sandbox
result = run_command_isolated(
    command=["ls", "-la"],
    working_dir="/path/to/repo",
    repo_path="/.repos/repo-name"
)

if result.success:
    print(result.stdout)
else:
    print(f"Error: {result.stderr}")
```

### Shell Command Usage

```python
from app.modules.utils.gvisor_runner import run_shell_command_isolated

# Run a shell command string
result = run_shell_command_isolated(
    shell_command="npm install",
    working_dir="/path/to/repo",
    timeout=300  # 5 minutes
)
```

### Check gVisor Availability

```python
from app.modules.utils.gvisor_runner import is_gvisor_available

if is_gvisor_available():
    print("gVisor is ready to use")
else:
    print("gVisor is not available, commands will run without isolation")
```

## How It Works

The gVisor runner uses the following approach:

1. **Primary Method**: Uses Docker with `runsc` runtime

   - Creates a temporary container using `busybox:latest` image
   - Mounts the working directory and repository paths
   - Executes the command in the isolated sandbox
   - Automatically cleans up the container after execution

2. **Fallback**: If Docker is not available, falls back to regular subprocess execution

## Requirements

- **Linux**: gVisor is primarily designed for Linux (kernel 4.14.77+)
- **Docker**: Recommended for best isolation (optional but recommended)
- **Architecture**: x86_64 or arm64

## Troubleshooting

### gVisor Installation Fails

- **Check architecture**: Ensure you're on a supported architecture (x86_64 or arm64)
- **Check permissions**: Installation may require sudo for system-wide installation
- **Check network**: Ensure you can download from `storage.googleapis.com`

### Docker Runtime Not Found

```bash
# Verify runsc is installed
which runsc

# Install as Docker runtime
sudo runsc install

# Reload Docker
sudo systemctl reload docker

# Verify runtime is available
docker info | grep -i runtime
```

### Commands Fail in gVisor

- **Check Docker**: Ensure Docker is running and runsc runtime is configured
- **Check mounts**: Verify the working directory and repo paths exist
- **Check logs**: Look at Docker logs for container errors
- **Fallback**: The system will automatically fall back to regular subprocess if gVisor fails

### Permission Issues

If you encounter permission issues:

```bash
# Add your user to docker group (if using Docker)
sudo usermod -aG docker $USER
# Log out and log back in for changes to take effect
```

## Development Workflow

1. **Install gVisor**: Run `python scripts/install_gvisor.py` or use manual installation
2. **Configure Docker**: Run `sudo runsc install` to integrate with Docker
3. **Test**: Verify with `docker run --rm --runtime=runsc busybox echo "test"`
4. **Use in code**: Import and use `run_command_isolated()` in your tools/agents

## Security Considerations

- gVisor provides application-level isolation, not full VM isolation
- Network is disabled by default for security (`--network=none`)
- Repository paths are mounted read-only when specified
- Containers are automatically removed after execution (`--rm`)

## Additional Resources

- [gVisor Documentation](https://gvisor.dev/docs/)
- [gVisor Installation Guide](https://gvisor.dev/docs/user_guide/install/)
- [Docker Runtime Configuration](https://docs.docker.com/engine/reference/commandline/dockerd/#daemon-runtime-options)
