"""Tests для memory.py — build_memory_block (surfacing персональной памяти)."""
from __future__ import annotations

from decimal import Decimal

from ayla_ai_core.memory import (
    INFERRED_MARK,
    MEMORY_BLOCK_HEADER,
    MEMORY_INFERRED_HEADER,
    SOURCE_INFERRED,
    SOURCE_STATED,
    build_memory_block,
)


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


# ---------------------------------------------------------------------------
# Происхождение факта (P0-3) — «сказал человек» против «вывела система».
# ---------------------------------------------------------------------------


def test_inferred_fact_is_distinguishable_in_the_rendered_block() -> None:
    """Выведенный факт НЕ должен выглядеть как сказанный человеком.

    Это и есть проверяемая граница: не «есть поле», а «различимо в тексте,
    который реально уходит в промпт».
    """
    out = build_memory_block(
        {"diet_type": "vegan", "busy_days": ["tue"]},
        sources={"diet_type": SOURCE_STATED, "busy_days": SOURCE_INFERRED},
    )
    stated_line = "- Диета: vegan"
    inferred_line = f"- {INFERRED_MARK} Избегает: вторник"
    assert stated_line in out
    assert inferred_line in out
    # Догадка живёт под заголовком-правилом, а сказанное — над ним.
    assert MEMORY_INFERRED_HEADER in out
    assert out.index(stated_line) < out.index(MEMORY_INFERRED_HEADER)
    assert out.index(MEMORY_INFERRED_HEADER) < out.index(inferred_line)
    # И ни один сказанный факт не носит метку вывода.
    assert INFERRED_MARK not in stated_line


def test_inferred_and_stated_are_not_the_same_string() -> None:
    """Один и тот же факт из двух источников рендерится по-разному."""
    stated = build_memory_block({"busy_days": ["tue"]}, sources={"busy_days": SOURCE_STATED})
    inferred = build_memory_block({"busy_days": ["tue"]}, sources={"busy_days": SOURCE_INFERRED})
    assert stated != inferred
    assert INFERRED_MARK in inferred and INFERRED_MARK not in stated
    assert MEMORY_INFERRED_HEADER in inferred and MEMORY_INFERRED_HEADER not in stated


def test_without_sources_output_is_byte_identical() -> None:
    """Отрицательный: уже верно помеченные факты не изменились.

    Ни один существующий вызывающий (бэкенд, старый бот) не передаёт
    ``sources`` — их блок обязан остаться прежним до байта.
    """
    ctx = {
        "preferred_time_slots": ["evening"],
        "diet_type": "vegan",
        "busy_days": ["tue"],
        "min_rating_preference": 4.8,
    }
    baseline = build_memory_block(ctx)
    assert build_memory_block(ctx, sources=None) == baseline
    assert build_memory_block(ctx, sources={}) == baseline
    # stated == «как раньше», и неизвестное значение тоже не ломает рендер.
    assert build_memory_block(ctx, sources=dict.fromkeys(ctx, SOURCE_STATED)) == baseline
    assert build_memory_block(ctx, sources={"diet_type": "whatever"}) == baseline
    assert INFERRED_MARK not in baseline


def test_provenance_is_orthogonal_to_confidence() -> None:
    """«Кто сказал» и «насколько уверены» не выводятся друг из друга."""
    out = build_memory_block(
        {"busy_days": ["tue"], "diet_type": "vegan"},
        confidences={"busy_days": 1.0, "diet_type": 0.5},
        sources={"busy_days": SOURCE_INFERRED, "diet_type": SOURCE_STATED},
    )
    # Уверенная догадка всё равно помечена как догадка…
    assert f"- {INFERRED_MARK} Избегает: вторник" in out
    # …а неуверенное утверждение человека смягчено, но НЕ помечено выводом.
    assert "- кажется, Диета: vegan" in out


def test_low_confidence_inferred_stays_in_clarify_without_mark() -> None:
    """<0.4 уходит в «уточнить» — это вопрос, его нельзя выдать за слова клиента."""
    out = build_memory_block(
        {"busy_days": ["tue"]},
        confidences={"busy_days": 0.2},
        sources={"busy_days": SOURCE_INFERRED},
    )
    assert "Стоит уточнить" in out
    assert INFERRED_MARK not in out
    assert MEMORY_INFERRED_HEADER not in out


def test_max_facts_caps_the_total_not_each_group() -> None:
    """Пометка происхождения не должна удваивать бюджет блока."""
    ctx = {
        "preferred_time_slots": ["evening"],
        "workplace_district": "Западная поляна",
        "busy_days": ["tue"],
        "min_rating_preference": 4.8,
        "diet_type": "vegan",
    }
    out = build_memory_block(
        ctx,
        sources={"busy_days": SOURCE_INFERRED, "diet_type": SOURCE_INFERRED},
        max_facts=2,
    )
    assert len([ln for ln in out.splitlines() if ln.startswith("- ")]) == 2


def test_derived_budget_is_marked_even_though_it_is_built_from_two_keys() -> None:
    """`price_range` рендерится из price_range_min/max — источник берётся у обоих."""
    ctx = {"price_range_min": Decimal("1000"), "price_range_max": Decimal("2500")}
    assert INFERRED_MARK in build_memory_block(
        ctx, sources={"price_range_min": SOURCE_INFERRED}
    )
    assert INFERRED_MARK in build_memory_block(
        ctx, sources={"price_range_max": SOURCE_INFERRED}
    )
    # И под собственным именем строки тоже — на случай, если источник придёт так.
    assert INFERRED_MARK in build_memory_block(ctx, sources={"price_range": SOURCE_INFERRED})
    assert INFERRED_MARK not in build_memory_block(
        ctx, sources={"price_range_min": SOURCE_STATED, "price_range_max": SOURCE_STATED}
    )
