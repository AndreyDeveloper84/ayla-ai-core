"""Tests для memory.py — build_memory_block (surfacing персональной памяти)."""
from __future__ import annotations

from decimal import Decimal

from ayla_ai_core.memory import MEMORY_BLOCK_HEADER, build_memory_block


def test_empty_context_returns_empty_string() -> None:
    assert build_memory_block({}) == ""
    # Пустые/None значения не создают фактов.
    assert build_memory_block({"preferred_time_slots": [], "diet_type": ""}) == ""


def test_basic_green_fields_rendered() -> None:
    out = build_memory_block(
        {
            "preferred_time_slots": ["evening"],
            "price_range_min": Decimal("1000"),
            "price_range_max": Decimal("2500"),
            "workplace_district": "Западная поляна",
            "busy_days": ["mon"],
            "min_rating_preference": 4.8,
        }
    )
    assert MEMORY_BLOCK_HEADER in out
    assert "вечер" in out
    assert "Бюджет 1000–2500 ₽" in out
    assert "рядом с работой (Западная поляна)" in out
    assert "понедельник" in out
    assert "4.8" in out


def test_favorite_masters_gives_id_and_name() -> None:
    out = build_memory_block(
        {"favorite_masters": ["abc-123"]},
        master_names={"abc-123": "Анна"},
    )
    assert "Анна" in out
    assert "id=abc-123" in out  # id всегда даётся LLM для tool-call


def test_favorite_masters_without_name_falls_back_to_id() -> None:
    out = build_memory_block({"favorite_masters": ["abc-123"]})
    assert "id=abc-123" in out


def test_confidence_high_asserts_medium_softens() -> None:
    high = build_memory_block(
        {"preferred_time_slots": ["evening"]},
        confidences={"preferred_time_slots": 0.95},
    )
    assert "кажется" not in high

    medium = build_memory_block(
        {"preferred_time_slots": ["evening"]},
        confidences={"preferred_time_slots": 0.5},
    )
    assert "кажется" in medium


def test_confidence_low_goes_to_clarify_not_facts() -> None:
    out = build_memory_block(
        {"preferred_time_slots": ["evening"]},
        confidences={"preferred_time_slots": 0.2},
    )
    assert "Стоит уточнить" in out
    # Низкая уверенность не утверждается как факт.
    assert "- Обычно выбирает время" not in out


def test_price_only_max() -> None:
    out = build_memory_block({"price_range_max": 3000})
    assert "Бюджет до 3000 ₽" in out


def test_forbidden_unknown_keys_ignored() -> None:
    # §2: неизвестные/чужие ключи (напр. потенциально red) не попадают в блок.
    out = build_memory_block({"pregnancy": True, "raw_chat_log": "..."})
    assert out == ""


def test_max_facts_caps_block() -> None:
    ctx = {
        "favorite_masters": ["m1"],
        "preferred_time_slots": ["evening"],
        "price_range_min": 1000,
        "price_range_max": 2000,
        "workplace_district": "A",
        "home_district": "B",
        "busy_days": ["mon"],
        "min_rating_preference": 4.5,
        "diet_type": "veg",
    }
    out = build_memory_block(ctx, max_facts=3)
    fact_lines = [ln for ln in out.splitlines() if ln.startswith("- ")]
    assert len(fact_lines) == 3
    # Приоритетные поля (favorite_masters, time) сохранены.
    assert any("Любимые мастера" in ln for ln in fact_lines)
