"""Tests for v0.7.3 / Obs-1 (DRF-681) + Obs-3 (DRF-683):
tenant-aware logging plumbing + replay frozen clock.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime

import pytest

from ayla_ai_core.observability import (
    ReplayDeterminismError,
    TenantContextFilter,
    current_frozen_now,
    current_tenant_id,
    reset_tenant_id,
    scope_frozen_now,
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


# ─── DRF-683 (v0.7.3 / Obs-3): replay frozen clock ────────────────────────


def test_frozen_now_default_is_none_outside_scope() -> None:
    """No scope -> None (vs tenant_id's empty-string default).

    None is correct here because consumer code must decide whether to fall
    back to wall-clock OR refuse to run; the empty-string convention used
    for tenant_id would force every renderer to special-case the value.
    """
    assert current_frozen_now() is None


def test_frozen_now_sync_scope_binds_and_restores() -> None:
    moment = datetime(2026, 5, 19, 12, 0, 0, tzinfo=UTC)
    with scope_frozen_now(moment):
        assert current_frozen_now() == moment
    assert current_frozen_now() is None


def test_frozen_now_nested_scope_overrides_outer() -> None:
    """Nested scope sets a new value; exiting restores the outer."""
    outer = datetime(2026, 5, 1, tzinfo=UTC)
    inner = datetime(2026, 5, 20, tzinfo=UTC)
    with scope_frozen_now(outer):
        assert current_frozen_now() == outer
        with scope_frozen_now(inner):
            assert current_frozen_now() == inner
        assert current_frozen_now() == outer
    assert current_frozen_now() is None


def test_frozen_now_none_inside_scope_explicitly_clears() -> None:
    """Passing None into a nested scope clears the clock — useful when a
    sub-step under a replay scope legitimately needs wall-clock."""
    outer = datetime(2026, 5, 1, tzinfo=UTC)
    with scope_frozen_now(outer), scope_frozen_now(None):
        assert current_frozen_now() is None


@pytest.mark.asyncio
async def test_frozen_now_concurrent_tasks_isolated() -> None:
    """ContextVar isolation across asyncio tasks — like tenant_id, two
    concurrent replay runs must not bleed into each other.
    """
    a = datetime(2026, 1, 1, tzinfo=UTC)
    b = datetime(2027, 1, 1, tzinfo=UTC)
    seen: dict[str, datetime | None] = {}

    async def worker(name: str, moment: datetime) -> None:
        async with scope_frozen_now(moment):
            await asyncio.sleep(0.01)
            seen[name] = current_frozen_now()

    await asyncio.gather(worker("a", a), worker("b", b))
    assert seen == {"a": a, "b": b}


def test_replay_determinism_error_subclasses_runtime_error() -> None:
    """ReplayDeterminismError is a RuntimeError subclass so consumers'
    generic except RuntimeError still catches it.
    """
    assert issubclass(ReplayDeterminismError, RuntimeError)
    err = ReplayDeterminismError("custom dispatcher called random.uniform")
    assert "custom dispatcher" in str(err)


@pytest.mark.asyncio
async def test_send_message_propagates_frozen_now_into_scope() -> None:
    """Integration: send_message(..., frozen_now=X) must bind X for the
    duration of prompt rendering + tool dispatch so consumer code that
    reads :func:`current_frozen_now` sees the value (and replay produces
    byte-identical output across runs).
    """
    import json
    from types import SimpleNamespace

    from ayla_ai_core.context import MasterCandidate, build_master_context_from_candidates
    from ayla_ai_core.orchestrator import AIConcierge

    seen_during_render: list[datetime | None] = []

    def capturing_renderer(_ctx):
        seen_during_render.append(current_frozen_now())
        return "system"

    candidates = [
        MasterCandidate(id=1, name="A", specialization="m", services=[(10, "s")]),
    ]
    ctx = build_master_context_from_candidates(candidates, tenant_id="t")

    class FakeStore:
        def resolve_active_conversation(self, _user_key):
            return SimpleNamespace(id=1)

        def save_message(self, conversation, **_kwargs):
            return SimpleNamespace(id=1)

        def load_recent_history(self, _conv, **_kwargs):
            return []

    class FakeClient:
        def __init__(self):
            self.chat = SimpleNamespace(completions=self)

        async def create(self, **_kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="ok", tool_calls=None),
                    ),
                ],
                usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
            )

    concierge = AIConcierge(
        openai_client=FakeClient(),
        store=FakeStore(),
        context_builder=lambda: ctx,
    )
    moment = datetime(2026, 5, 19, 12, 0, 0, tzinfo=UTC)
    await concierge.send_message(
        user_key=1,
        message_text="hi",
        prompt_renderer=capturing_renderer,
        frozen_now=moment,
    )

    assert seen_during_render == [moment]
    # And the scope is released after the call returns.
    assert current_frozen_now() is None
    # json import was unused locally; keep linter quiet by referencing it.
    _ = json
