# Memory System Setup

## Prerequisites

**⚠️ Important**: The memory system requires the **Letta server** to be running, which needs an **OpenAI API key** in your environment.

Before using the memory features, ensure:
1. `OPENAI_API_KEY` is set in your environment
2. Letta server is running (starts automatically with `docker-compose up`)

## Quick Start

### 1. Set Environment Variables

```bash
export OPENAI_API_KEY=your_openai_api_key_here
```

### 2. Start Letta Server

The Letta server runs as a Docker container. Start it with:

```bash
docker-compose up -d letta
```

Or start all services:

```bash
docker-compose up -d
```

### 3. Verify Letta is Running

```bash
# Check if the server is responding
curl http://localhost:8283/

# Or check container logs
docker logs potpie-letta
```

The Letta web UI is available at **http://localhost:8283** for viewing agents and memories.

## Configuration

Memory system configuration via environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | **Required** - OpenAI API key for LLM and embeddings | - |
| `LETTA_SERVER_URL` | Letta server URL | `http://localhost:8283` |
| `LETTA_MODEL` | LLM model in `provider/model` format | `openai/gpt-4o-mini` |
| `LETTA_EMBEDDING` | Embedding model | `openai/text-embedding-ada-002` |

## How It Works

Potpie uses **[Letta](https://www.letta.com/)** for stateful agent memory management. The system:

1. **Extracts facts** from conversations using LLM-based fact extraction
2. **Categorizes memories** into:
   - **User-level**: Personal preferences that persist across all projects
   - **Project-level**: Project-specific information scoped to individual codebases
   - **Memory types**: Semantic (general knowledge) or Episodic (events/actions)
3. **Stores passages** in Letta for semantic search
4. **Retrieves relevant memories** when needed using embedding-based search

## Key Features

- ✅ **User-level memories**: Personal preferences, coding styles, tool preferences
- ✅ **Project-level memories**: Tech stack, architecture decisions, project-specific context
- ✅ **Semantic search**: Intelligent memory retrieval using embeddings
- ✅ **Automatic fact extraction**: LLM extracts meaningful facts from conversations
- ✅ **Memory categorization**: Automatically determines scope (user vs project) and type

## Architecture

```
┌─────────────────┐
│  Conversation   │
│     Service     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  LettaService   │
│  (Memory Layer) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Letta Server   │
│  (Docker)        │
│  Port: 8283     │
└─────────────────┘
```

## API Usage

The memory system is accessed through:
- **Memory Router**: `/api/memories` endpoints
- **Memory Service Factory**: Creates `LettaService` instances
- **Memory Search Tool**: Integrated into agent toolkits

## Troubleshooting

**Letta server not responding?**
- Check if container is running: `docker ps | grep letta`
- Check logs: `docker logs potpie-letta`
- Verify `OPENAI_API_KEY` is set in the container environment

**Memory not being stored?**
- Ensure Letta server is healthy (check healthcheck)
- Verify OpenAI API key is valid
- Check application logs for memory service errors

**Can't connect to Letta?**
- Verify `LETTA_SERVER_URL` matches the running container
- Check network connectivity between services
- Ensure port 8283 is accessible

## Related Files

- **Service Implementation**: `app/modules/intelligence/memory/letta_service.py`
- **Service Factory**: `app/modules/intelligence/memory/memory_service_factory.py`
- **API Router**: `app/modules/intelligence/memory/memory_router.py`
- **Docker Config**: `docker-compose.yaml` (letta service)
