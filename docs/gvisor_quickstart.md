# gVisor Quick Start for Linux Development

## Quick Setup (3 Steps)

### 1. Install gVisor

```bash
# Automatic installation (recommended)
python scripts/install_gvisor.py

# Or manual installation
sudo apt-get update && sudo apt-get install -y runsc
```

### 2. Configure Docker Runtime

```bash
# Install runsc as Docker runtime
sudo runsc install

# Reload Docker
sudo systemctl reload docker

# Verify it works
docker run --rm --runtime=runsc busybox echo "Hello from gVisor"
```

### 3. Use in Your Code

```python
from app.modules.utils.gvisor_runner import run_command_isolated

result = run_command_isolated(
    command=["ls", "-la"],
    working_dir="/path/to/repo",
    repo_path="/.repos/repo-name"
)
```

## That's It!

The system will automatically:
- ✅ Use gVisor when Docker + runsc runtime is available
- ✅ Fall back to regular subprocess if gVisor is not available
- ✅ Isolate commands in sandboxed containers
- ✅ Clean up containers after execution

## Troubleshooting

**Problem**: `docker: Error response from daemon: Unknown runtime specified runsc`

**Solution**:
```bash
sudo runsc install
sudo systemctl reload docker
```

**Problem**: `runsc: command not found`

**Solution**:
```bash
python scripts/install_gvisor.py
# Or install manually to /usr/local/bin
```

For more details, see [gvisor_setup.md](./gvisor_setup.md)
