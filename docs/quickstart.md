# Quickstart Guide

These steps will get you up and running with :contentReference[oaicite:0]{index=0} in under 5 minutes.

```bash
# 1. Clone the repository
git clone https://github.com/potpie-ai/potpie.git
cd potpie

# 2. Copy environment template
cp .env.template .env

# 3. Install dependencies
uv sync

# 4. Start services (Linux / macOS)
chmod +x start.sh
./start.sh

# 4b. On Windows (PowerShell)
# ./start.ps1

# 5. Open the UI
# Once startup is complete, visit:
http://localhost:8001
```

**That’s it.** You should now have Potpie running locally.

If you need database migrations, Neo4j, Redis, or other services, `start.sh` will take care of it automatically.  

For a full setup, see the detailed instructions in [Setup Guide](./setup.md).
