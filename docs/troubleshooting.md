# Troubleshooting & Common Issues

## Common Setup Issues

### ❗ Environment Variables Missing or Incorrect
- Ensure you copied `.env.template` to `.env` and filled all required fields.  
- Double-check values like `NEO4J_URI`, `NEO4J_PASSWORD`, `REDISHOST`, `POSTGRES_SERVER`.

### ❗ Python / Dependencies Problems
- Potpie requires **Python 3.11+**.  
- Ensure `uv sync` has run without errors and `~/.local/bin` is in your PATH (so that `uv` is executable).  
- If dependencies fail, check whether you have correct system-level libraries (e.g. `libssl`, `build-essential`, etc.).

## Runtime / Server Issues

### ❗ Neo4j Connection Failed
- Common cause: wrong password, port or URI.  
- Make sure Neo4j service is running and matching `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD` in `.env`.

### ❗ Redis or PostgreSQL Not Running
- Ensure Docker (or local services) are running before `start.sh`.  
- Check that ports `6379` (Redis) and default Postgres port are free.

### ❗ Parsing Fails or Agents Return Empty Results
- Make sure the repository path you provided exists and is accessible.  
- Ensure Git credentials or access tokens are correct.  
- Check logs for errors — sometimes external LLM provider API errors cause failures.  

## Helpful Debugging Commands

```bash
# Check status of services
docker ps

# View backend logs
docker-compose logs backend

# Re-run parsing
curl -X POST 'http://localhost:8001/api/v1/parse' \
     -H 'Content-Type: application/json' \
     -d '{"repo_path":"path/to/your/repo", "branch_name":"main"}'
```

If nothing helps — open an issue or check the documentation section on configuration.  