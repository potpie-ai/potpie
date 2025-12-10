# Architecture Overview

## Major Components

- **Frontend UI** – user interface (React / web)  
- **Backend API** – service layer (FastAPI) handling requests  
- **Database** – PostgreSQL for persistence  
- **Knowledge Graph** – Neo4j storing relationships between code entities  
- **Queue / Worker System** – Redis + Celery for background tasks  
- **Agent Layer** – LLM-based agents that query the Graph and codebase  
- **Repository Parser** – extracts structure from Git repos and populates the Graph  
