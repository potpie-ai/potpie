from __future__ import annotations

from typing import Any

_KNOWN_CORRELATION_KEYS = (
    "conversation_id",
    "run_id",
    "project_id",
    "user_id",
    "request_id",
)


def install_celery_observability(celery_app: Any = None, config: Any = None) -> None:
    del celery_app
    from celery.signals import task_postrun, task_prerun, worker_process_init

    from observability import _pop_context, _push_context, configure
    from observability.profiles import celery as celery_profile

    cfg = config or celery_profile()
    pending: dict[Any, Any] = {}

    @worker_process_init.connect
    def _init_worker(**_kwargs):
        configure(cfg)

    @task_prerun.connect
    def _on_prerun(task_id=None, task=None, kwargs=None, **_kwargs):
        fields = {"task_id": task_id, "task_name": getattr(task, "name", None)}
        if isinstance(kwargs, dict):
            for key in _KNOWN_CORRELATION_KEYS:
                if key in kwargs:
                    fields[key] = kwargs[key]
        pending[task_id] = _push_context(**fields)

    @task_postrun.connect
    def _on_postrun(task_id=None, **_kwargs):
        token = pending.pop(task_id, None)
        if token is not None:
            _pop_context(token)
