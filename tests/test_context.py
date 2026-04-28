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
