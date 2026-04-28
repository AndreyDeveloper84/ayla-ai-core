"""Tests для context.py — MasterCandidate / MasterContext / render_summary_text."""
from __future__ import annotations

from ayla_ai_core.context import (
    MasterCandidate,
    MasterContext,
    build_master_context_from_candidates,
    render_summary_text,
)

# ─── render_summary_text ──────────────────────────────────────────────────


def test_render_summary_empty_candidates() -> None:
    assert "нет активных мастеров" in render_summary_text([])


def test_render_summary_one_candidate_with_services() -> None:
    candidates = [
        MasterCandidate(
            id=42,
            name="Анна Иванова",
            specialization="массаж",
            services=[(10, "массаж спины"), (11, "лимфодренаж")],
        ),
    ]
    out = render_summary_text(candidates)
    assert "master_id=42" in out
    assert "Анна Иванова" in out
    assert "(массаж)" in out
    assert "service_id=10 массаж спины" in out
    assert "service_id=11 лимфодренаж" in out


def test_render_summary_caps_at_5_services_with_overflow_note() -> None:
    services = [(i, f"услуга {i}") for i in range(1, 9)]  # 8 услуг
    candidates = [MasterCandidate(id=1, name="X", specialization="", services=services)]
    out = render_summary_text(candidates)
    # First 5 — отображены
    for i in range(1, 6):
        assert f"service_id={i}" in out
    # 6+ — нет, но есть overflow note
    assert "service_id=6" not in out
    assert "+ещё 3" in out


def test_render_summary_no_specialization_no_parens() -> None:
    candidates = [MasterCandidate(id=1, name="X", specialization="", services=[])]
    out = render_summary_text(candidates)
    assert "(" not in out  # нет пустых скобок


# ─── build_master_context_from_candidates ─────────────────────────────────


def test_build_context_collects_all_service_ids_across_masters() -> None:
    candidates = [
        MasterCandidate(id=1, name="A", specialization="", services=[(10, "s1"), (11, "s2")]),
        MasterCandidate(id=2, name="B", specialization="", services=[(11, "s2"), (12, "s3")]),
    ]
    ctx = build_master_context_from_candidates(candidates)
    assert ctx.candidate_ids == frozenset({1, 2})
    assert ctx.candidate_service_ids == frozenset({10, 11, 12})  # дедуп через set
    assert ctx.summary_text  # не пустой


def test_build_context_empty_returns_empty_frozensets() -> None:
    ctx = build_master_context_from_candidates([])
    assert ctx.candidate_ids == frozenset()
    assert ctx.candidate_service_ids == frozenset()
    assert "нет активных мастеров" in ctx.summary_text


# ─── MasterContext immutability (frozen dataclass) ────────────────────────


def test_master_context_is_frozen() -> None:
    ctx = MasterContext(
        candidates=[],
        candidate_ids=frozenset(),
        candidate_service_ids=frozenset(),
        summary_text="x",
    )
    import pytest

    with pytest.raises(AttributeError):
        ctx.summary_text = "y"  # type: ignore[misc]


# ─── DRF-238: SpecialistContext generic + tenant_id ───────────────────────


def test_specialist_context_alias_for_master_context() -> None:
    """MasterContext / MasterCandidate — backward compat aliases для бота."""
    from ayla_ai_core.context import (
        MasterCandidate,
        MasterContext,
        SpecialistCandidate,
        SpecialistContext,
    )
    # Generic specializations equal aliases
    assert MasterCandidate is SpecialistCandidate[int]
    assert MasterContext is SpecialistContext[int]


def test_specialist_context_with_uuid_ids() -> None:
    """SpecialistContext[UUID] для Ayla — UUID IDs работают."""
    from uuid import UUID

    from ayla_ai_core.context import (
        SpecialistCandidate,
        build_specialist_context_from_candidates,
    )

    uid_a = UUID("11111111-1111-1111-1111-111111111111")
    uid_b = UUID("22222222-2222-2222-2222-222222222222")
    sid_x = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")

    candidates = [
        SpecialistCandidate(id=uid_a, name="Анна", specialization="массаж", services=[(sid_x, "массаж")]),
        SpecialistCandidate(id=uid_b, name="Борис", specialization="спа", services=[(sid_x, "массаж")]),
    ]
    ctx = build_specialist_context_from_candidates(candidates)
    assert uid_a in ctx.candidate_ids
    assert sid_x in ctx.candidate_service_ids
    # summary_text использует str(c.id) — UUIDs render-friendly
    assert "11111111-1111" in ctx.summary_text


def test_specialist_context_with_tenant_id() -> None:
    """tenant_id передаётся для multi-tenant scoping (Ayla)."""
    from ayla_ai_core.context import build_specialist_context_from_candidates

    ctx = build_specialist_context_from_candidates([], tenant_id="formula-tela")
    assert ctx.tenant_id == "formula-tela"


def test_specialist_context_default_tenant_is_none() -> None:
    """Single-tenant случай (бот) — tenant_id остаётся None."""
    from ayla_ai_core.context import build_specialist_context_from_candidates

    ctx = build_specialist_context_from_candidates([])
    assert ctx.tenant_id is None


def test_master_context_helper_no_tenant_id_param() -> None:
    """Backward compat helper не принимает tenant_id (бот single-tenant)."""
    from ayla_ai_core.context import build_master_context_from_candidates

    ctx = build_master_context_from_candidates([])
    # tenant_id остаётся None для бота
    assert ctx.tenant_id is None
