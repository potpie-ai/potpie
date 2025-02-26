# Development Guidelines for Momentum Server

## Build & Run Commands
- Setup: `python3.10 -m venv venv && source venv/bin/activate`
- Install: `pip install -r requirements.txt`
- Start server: `chmod +x start.sh && ./start.sh`
- Lint: `black . && ruff check . && isort .`
- Typecheck: `mypy .`
- Run tests: `pytest tests/`
- Run single test: `pytest tests/path/to/test_file.py::test_name`

## Code Style Guidelines
- **Python/FastAPI**: Functional, declarative programming; avoid classes where possible
- **Naming**: Descriptive with auxiliary verbs (is_active, has_permission); lowercase with underscores
- **Types**: Use type hints for all function signatures; prefer Pydantic models for validation
- **Error handling**: Handle errors at beginning of functions; use early returns and guard clauses
- **Formatting**: Run `black`, `ruff` and `isort` before committing
- **File structure**: Router exports, sub-routes, utilities, static content, types/models
- **Dependencies**: FastAPI, Pydantic v2, SQLAlchemy 2.0, async database libraries
- **Performance**: Use async functions for I/O-bound tasks; implement caching for static data
- **Routes**: Use dependency injection for shared resources; keep routes and dependencies clear