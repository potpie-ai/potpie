# Configuring Docker Desktop to Use gVisor

## Status

✅ **runsc is installed** in Docker Desktop's VM at `/usr/local/bin/runsc`

## Next Steps: Configure Docker Desktop

### Option 1: Using Docker Desktop GUI (Recommended)

1. **Open Docker Desktop**
2. **Go to Settings** (gear icon in top right)
3. **Click "Docker Engine"** in the left sidebar
4. **Edit the JSON configuration** - add the runsc runtime:

```json
{
  "runtimes": {
    "runsc": {
      "path": "/usr/local/bin/runsc"
    }
  }
}
```

5. **Click "Apply & Restart"**
6. **Wait for Docker Desktop to restart**

### Option 2: Using Command Line

The daemon.json file is already configured at `~/.docker/daemon.json`, but Docker Desktop may not use it directly. You still need to configure it through the GUI.

## Verify Setup

After restarting Docker Desktop, run:

```bash
# Check if runsc runtime is available
docker info --format "{{.Runtimes}}" | grep runsc

# Test gVisor
docker run --rm --runtime=runsc busybox echo "Hello from gVisor"
```

## If It Doesn't Work

If the runtime doesn't appear:

1. **Check Docker Desktop Settings** - Make sure the runtime is configured in the GUI
2. **Verify runsc is in the VM**:
   ```bash
   docker run --rm alpine ls -la /usr/local/bin/runsc
   ```
3. **Try restarting Docker Desktop again**
4. **Check Docker Desktop logs** for any errors

## Testing with Your Code

Once configured, your code will automatically detect and use gVisor:

```python
from app.modules.utils.gvisor_runner import is_gvisor_available, run_command_isolated

# Check if gVisor is available
print(f"gVisor available: {is_gvisor_available()}")  # Should be True

# Use it
result = run_command_isolated(
    command=["echo", "Hello"],
    use_gvisor=True
)
```

