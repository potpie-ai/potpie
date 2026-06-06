"""Built-in sinks. Each conforms to observability.sink.Sink.

Sinks are registered lazily (their optional deps may be absent) — importing
this package must NOT import loguru/sentry_sdk/logfire. Resolution-time only.
"""
