"""Optional framework integrations. Imported lazily — the core never depends
on FastAPI/Starlette or Celery. Each integration re-binds correlation context
at its boundary (EC3: contextvars don't cross thread/process/task hops).
"""
