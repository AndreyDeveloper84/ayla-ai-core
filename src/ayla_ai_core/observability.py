"""Observability helpers — tenant-aware logging plumbing (v0.7.3 / Obs-1).

The library emits structured logs from several modules (:mod:`orchestrator`,
:mod:`tool_handlers`). In a multi-tenant deployment, operators need to filter
incidents to a single tenant; without a stable ``tenant_id`` field on every
record, "show me only the formula-tela errors" is impossible — incident
affecting one tenant pollutes the log stream for all.

This module provides a :class:`contextvars.ContextVar`-backed tenant scope
bound for the duration of one ``AIConcierge.send_message`` call. Every log
call inside the library reads the bound value via :func:`current_tenant_id`
and attaches it to ``LogRecord.extra``. The ContextVar approach is async-safe
out of the box — concurrent ``send_message`` calls in the same process
don't see each other's tenant.

Consumers can either:

1. **Pass-through structured logs**: every library log record already carries
   ``record.tenant_id``. Use a JSON formatter that emits the field:

   .. code-block:: python

       import json, logging
       class JsonFormatter(logging.Formatter):
           def format(self, record):
               payload = {
                   "msg": record.getMessage(),
                   "level": record.levelname,
                   "logger": record.name,
                   "tenant_id": getattr(record, "tenant_id", ""),
               }
               return json.dumps(payload)

2. **Filter pattern** (when your formatter expects ``%(tenant_id)s`` but
   third-party code calls ``logging`` without ``extra=``): install
   :class:`TenantContextFilter` on the ``ayla_ai_core`` logger so absent
   ``record.tenant_id`` is auto-populated from the ContextVar.

The ContextVar default is the empty string so logs emitted **outside** any
``send_message`` scope (test imports, library startup) still produce a
well-formed record.
"""
from __future__ import annotations

import contextvars
import logging

__all__ = [
    "TenantContextFilter",
    "current_tenant_id",
    "reset_tenant_id",
    "scope_tenant_id",
    "set_tenant_id",
]


# The default empty string is deliberate. None would force every consumer
# formatter to guard against `None`, and changing the type later is
# semver-painful. Empty string is filter-friendly ("if record.tenant_id" is
# false) and JSON-safe.
_tenant_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "ayla_ai_core.tenant_id", default=""
)


def current_tenant_id() -> str:
    """Return the tenant_id bound to the current execution context.

    Returns the empty string when no scope is active (e.g., library code
    invoked outside :meth:`AIConcierge.send_message`). Library log calls
    pass this through ``extra={"tenant_id": current_tenant_id()}`` so
    every record carries the field, even when the value is unknown.
    """
    return _tenant_id_var.get()


def set_tenant_id(value: str) -> contextvars.Token[str]:
    """Bind ``tenant_id`` to the current context; return reset Token.

    The caller is responsible for resetting via :func:`reset_tenant_id`
    once the scoped work is done — typically via ``try / finally`` so the
    token is released even when the body raises. Prefer
    :func:`scope_tenant_id` for new code; this low-level pair is exposed
    for callers who need explicit token control (e.g., context managers
    that span multiple awaits with their own try/finally bookkeeping).
    """
    return _tenant_id_var.set(value)


def reset_tenant_id(token: contextvars.Token[str]) -> None:
    """Release a scope opened by :func:`set_tenant_id`."""
    _tenant_id_var.reset(token)


class _Scope:
    """Sync + async context manager wrapper around set/reset."""

    __slots__ = ("_token", "_value")

    def __init__(self, value: str) -> None:
        self._value = value
        self._token: contextvars.Token[str] | None = None

    def __enter__(self) -> str:
        self._token = _tenant_id_var.set(self._value)
        return self._value

    def __exit__(self, *_exc: object) -> None:
        assert self._token is not None
        _tenant_id_var.reset(self._token)
        self._token = None

    async def __aenter__(self) -> str:
        return self.__enter__()

    async def __aexit__(self, *_exc: object) -> None:
        self.__exit__(*_exc)


def scope_tenant_id(value: str) -> _Scope:
    """Bind ``tenant_id`` for the duration of a ``with``/``async with`` block.

    Use this in :meth:`AIConcierge.send_message` so every log call emitted
    while the LLM round trip + tool dispatch run sees the right tenant —
    including library internals that don't take ``tenant_id`` as a param.

    Example::

        async with scope_tenant_id(context.tenant_id):
            completion = await openai_client.chat.completions.create(...)
            result = dispatch_tool_call(...)
    """
    return _Scope(value)


class TenantContextFilter(logging.Filter):
    """Auto-populate ``record.tenant_id`` from the ContextVar.

    Install this on the ``ayla_ai_core`` logger when your structured
    formatter expects ``%(tenant_id)s`` and you want third-party code paths
    (libraries logging without ``extra=``) to still produce a well-formed
    record. The filter is a no-op when the record already has the field
    (library log calls always pass it via ``extra=``).
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "tenant_id"):
            record.tenant_id = current_tenant_id()
        return True
