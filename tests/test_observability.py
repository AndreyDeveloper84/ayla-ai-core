"""Tests for v0.7.3 / Obs-1 (DRF-681): tenant-aware logging plumbing."""
from __future__ import annotations

import asyncio
import logging

import pytest

from ayla_ai_core.observability import (
    TenantContextFilter,
    current_tenant_id,
    reset_tenant_id,
    scope_tenant_id,
    set_tenant_id,
)


def test_default_is_empty_string_outside_scope() -> None:
    """No active scope -> empty string (not None). JSON/format-safe."""
    assert current_tenant_id() == ""


def test_set_and_reset_round_trip() -> None:
    """Low-level set/reset pair restores prior state."""
    assert current_tenant_id() == ""
    token = set_tenant_id("t1")
    try:
        assert current_tenant_id() == "t1"
    finally:
        reset_tenant_id(token)
    assert current_tenant_id() == ""


def test_scope_context_manager_sync() -> None:
    """`with scope_tenant_id(...)` binds for the body, restores on exit."""
    with scope_tenant_id("t1") as bound:
        assert bound == "t1"
        assert current_tenant_id() == "t1"
    assert current_tenant_id() == ""


def test_nested_scopes_restore_outer_value() -> None:
    """Nested scopes restore the enclosing scope, not the global default."""
    with scope_tenant_id("outer"):
        assert current_tenant_id() == "outer"
        with scope_tenant_id("inner"):
            assert current_tenant_id() == "inner"
        assert current_tenant_id() == "outer"
    assert current_tenant_id() == ""


@pytest.mark.asyncio
async def test_async_scope_works_across_await() -> None:
    """`async with scope_tenant_id(...)` survives an await boundary."""
    async with scope_tenant_id("async-t"):
        assert current_tenant_id() == "async-t"
        await asyncio.sleep(0)
        assert current_tenant_id() == "async-t"
    assert current_tenant_id() == ""


@pytest.mark.asyncio
async def test_concurrent_tasks_have_independent_scopes() -> None:
    """ContextVar isolation: parallel tasks see their own tenant_id.

    The whole reason for a ContextVar (not a module-global) is that two
    `send_message` calls running concurrently in the same process — one
    serving tenant A, one tenant B — must not leak tenant_id into each
    other's log records.
    """
    seen: dict[str, str] = {}

    async def worker(name: str, tenant: str) -> None:
        async with scope_tenant_id(tenant):
            await asyncio.sleep(0.01)
            seen[name] = current_tenant_id()

    await asyncio.gather(
        worker("a", "tenant-a"),
        worker("b", "tenant-b"),
    )

    assert seen == {"a": "tenant-a", "b": "tenant-b"}


def test_filter_populates_missing_tenant_id_on_record() -> None:
    """TenantContextFilter auto-fills record.tenant_id from ContextVar.

    Third-party code paths can call `logger.info(...)` without `extra=`;
    the filter ensures the structured formatter still sees the field.
    """
    f = TenantContextFilter()
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname=__file__, lineno=1,
        msg="hi", args=(), exc_info=None,
    )
    assert not hasattr(record, "tenant_id")

    with scope_tenant_id("filtered-tenant"):
        kept = f.filter(record)

    assert kept is True
    assert record.tenant_id == "filtered-tenant"


def test_filter_does_not_overwrite_explicit_extra() -> None:
    """If the caller passed extra={"tenant_id": ...} explicitly, the
    filter must respect it — library log calls (which always pass it
    via extra=) shouldn't be silently overridden by an empty scope."""
    f = TenantContextFilter()
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname=__file__, lineno=1,
        msg="hi", args=(), exc_info=None,
    )
    record.tenant_id = "explicit-value"

    # No active scope, but filter must keep the explicit value.
    kept = f.filter(record)

    assert kept is True
    assert record.tenant_id == "explicit-value"


def test_library_log_call_in_tool_handler_carries_tenant_id(caplog) -> None:
    """Integration: a fallback in _fallback_clarification logs a WARNING
    record carrying record.tenant_id set to the active scope value.
    Verifies the library-side wiring of Obs-1 (DRF-681).
    """
    from ayla_ai_core.tool_handlers import _fallback_clarification

    with (
        caplog.at_level(logging.WARNING, logger="ayla_ai_core.tool_handlers"),
        scope_tenant_id("integration-tenant"),
    ):
        _fallback_clarification("integration_test")

    matching = [r for r in caplog.records if r.message.startswith("ai.tool_call.fallback")]
    assert matching, "expected a WARNING from _fallback_clarification"
    assert matching[0].tenant_id == "integration-tenant"
