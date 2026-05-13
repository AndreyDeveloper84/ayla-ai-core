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
    ctx = build_master_context_from_candidates(candidates, tenant_id="test-tenant")
    assert ctx.candidate_ids == frozenset({1, 2})
    assert ctx.candidate_service_ids == frozenset({10, 11, 12})  # дедуп через set
    assert ctx.summary_text  # не пустой


def test_build_context_empty_returns_empty_frozensets() -> None:
    ctx = build_master_context_from_candidates([], tenant_id="test-tenant")
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
        tenant_id="test-tenant",
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
    ctx = build_specialist_context_from_candidates(candidates, tenant_id="test-tenant")
    assert uid_a in ctx.candidate_ids
    assert sid_x in ctx.candidate_service_ids
    # summary_text использует str(c.id) — UUIDs render-friendly
    assert "11111111-1111" in ctx.summary_text


def test_specialist_context_with_tenant_id() -> None:
    """tenant_id передаётся для multi-tenant scoping (Ayla)."""
    from ayla_ai_core.context import build_specialist_context_from_candidates

    ctx = build_specialist_context_from_candidates([], tenant_id="formula-tela")
    assert ctx.tenant_id == "formula-tela"


def test_specialist_context_requires_tenant_id_kwarg() -> None:
    """v0.7.0 BREAKING: tenant_id is mandatory — empty/missing raises.

    Was in v0.6.x: defaulted to None for single-tenant cases. Removed because
    multi-tenant isolation is a security boundary, not a feature flag.
    Consumers pass a stable sentinel (e.g., "formula-tela") even in single-tenant.
    """
    import pytest

    from ayla_ai_core.context import build_specialist_context_from_candidates

    with pytest.raises(ValueError, match="tenant_id is required"):
        build_specialist_context_from_candidates([], tenant_id="")
    with pytest.raises(TypeError):
        # tenant_id is kwarg-only; positional should fail
        build_specialist_context_from_candidates([])  # type: ignore[call-arg]


def test_master_context_helper_also_requires_tenant_id() -> None:
    """v0.7.0 BREAKING: deprecated helper also gates on tenant_id."""
    import pytest

    from ayla_ai_core.context import build_master_context_from_candidates

    with pytest.raises(ValueError, match="tenant_id is required"):
        build_master_context_from_candidates([], tenant_id="")


# ─── DRF-677 (v0.7.2 Perf-1): O(1) lookups via by_id + service_id_set ─────


def test_builder_populates_by_id_for_o1_handler_lookups() -> None:
    """v0.7.2: SpecialistContext.by_id maps {id -> candidate} after build."""
    candidates = [
        MasterCandidate(id=1, name="Anna", specialization="massage", services=[(10, "back")]),
        MasterCandidate(id=2, name="Boris", specialization="manicure", services=[(11, "nails")]),
    ]
    ctx = build_master_context_from_candidates(candidates, tenant_id="t")
    assert set(ctx.by_id.keys()) == {1, 2}
    assert ctx.by_id[1].name == "Anna"
    assert ctx.by_id[2].name == "Boris"


def test_builder_populates_service_id_set_per_candidate() -> None:
    """v0.7.2: each SpecialistCandidate carries a frozenset of its service IDs."""
    candidates = [
        MasterCandidate(
            id=1, name="Anna", specialization="massage",
            services=[(10, "back"), (11, "lymph")],
        ),
    ]
    ctx = build_master_context_from_candidates(candidates, tenant_id="t")
    assert ctx.candidates[0].service_id_set == frozenset({10, 11})


def test_direct_constructor_auto_populates_by_id_post_init() -> None:
    """v0.7.2: direct SpecialistContext(...) keeps backward-compat.

    Callers that bypass the builder don't have to compute by_id themselves;
    __post_init__ auto-populates from candidates when by_id is the default
    empty dict.
    """
    from ayla_ai_core.context import SpecialistContext

    candidates = [
        MasterCandidate(id=1, name="A", specialization="m", services=[(10, "s")]),
    ]
    ctx = SpecialistContext(
        candidates=candidates,
        candidate_ids=frozenset({1}),
        candidate_service_ids=frozenset({10}),
        summary_text="",
        tenant_id="t",
        # by_id intentionally omitted — defaults to {} then auto-populates.
    )
    assert ctx.by_id == {1: candidates[0]}


def test_direct_candidate_auto_populates_service_id_set_post_init() -> None:
    """v0.7.2: direct SpecialistCandidate(...) keeps backward-compat too."""
    c = MasterCandidate(
        id=1, name="A", specialization="m", services=[(10, "s"), (11, "t")]
    )
    assert c.service_id_set == frozenset({10, 11})
