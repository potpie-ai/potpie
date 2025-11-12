# gVisor Usage Guide

## How It Works in Different Environments

### 1. Kubernetes (K8s) / Container Environments

**Setup**: gVisor is automatically installed in the Docker image.

**How it works**:
- The system detects it's running in a container
- Uses `runsc` directly (installed at `/usr/local/bin/runsc`)
- Provides additional isolation for commands executed within the container
- Falls back to regular subprocess if runsc isn't available

**No configuration needed** - it just works!

### 2. Local Linux Development

**Setup**: Optional - install gVisor and configure Docker runtime.

**How it works**:
- If Docker + runsc runtime is configured: Uses Docker with gVisor
- Otherwise: Falls back to regular subprocess

**Setup steps** (optional):
```bash
# Install gVisor
python scripts/install_gvisor.py

# Configure Docker runtime
sudo runsc install
sudo systemctl reload docker
```

### 3. Local Mac/Windows Development

**Setup**: Not needed - automatically uses fallback.

**How it works**:
- Detects non-Linux platform
- Automatically uses regular subprocess (gVisor not supported on Mac/Windows)
- No configuration needed

## Usage in Code

The API is the same regardless of environment:

```python
from app.modules.utils.gvisor_runner import run_command_isolated

# Run a command - automatically uses gVisor if available
result = run_command_isolated(
    command=["npm", "install"],
    working_dir="/path/to/repo",
    repo_path="/.repos/repo-name",
    timeout=300
)

if result.success:
    print(result.stdout)
else:
    print(f"Error: {result.stderr}")
```

## Environment Detection

The system automatically detects:

1. **Platform**: Linux vs Mac/Windows
2. **Container**: Running in Docker/K8s vs host
3. **Docker**: Docker available with runsc runtime
4. **runsc**: runsc binary available

Based on these, it chooses the best method:
- ✅ K8s + runsc: Use runsc directly
- ✅ Linux + Docker + runsc: Use Docker with runsc runtime
- ✅ Mac/Windows: Use regular subprocess
- ✅ Fallback: Use regular subprocess if gVisor fails

## Benefits

- **K8s**: Additional isolation layer for commands within containers
- **Local Linux**: Full gVisor isolation when configured
- **Local Mac/Windows**: Works seamlessly without gVisor
- **Automatic**: No code changes needed - works everywhere

## Troubleshooting

### In K8s

**Q: Is gVisor working in my K8s pods?**
```bash
# Check if runsc is installed
kubectl exec <pod-name> -- runsc --version

# Check logs for gVisor usage
kubectl logs <pod-name> | grep -i gvisor
```

**Q: Commands fail with gVisor errors?**
- The system will automatically fall back to regular subprocess
- Check pod security context - may need additional permissions
- Container isolation still provides security even without gVisor

### Local Development

**Q: How do I know if gVisor is being used?**
```python
from app.modules.utils.gvisor_runner import is_gvisor_available
print(f"gVisor available: {is_gvisor_available()}")
```

**Q: Commands work but gVisor isn't being used?**
- On Mac/Windows: This is expected - gVisor isn't supported
- On Linux: Check Docker + runsc runtime configuration
- The fallback to regular subprocess is automatic and safe

## Security Considerations

- **K8s**: Container + gVisor provides defense in depth
- **Local**: gVisor adds extra isolation when configured
- **Fallback**: Regular subprocess is still secure for local development
- **Network**: Commands run with network disabled when using gVisor
