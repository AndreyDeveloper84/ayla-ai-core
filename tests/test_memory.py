"""Tests для memory.py — build_memory_block (surfacing персональной памяти)."""
from __future__ import annotations

from decimal import Decimal

import pytest

from ayla_ai_core.memory import (
    _CONTEXT_KEY_TO_FIELD,
    _FIELD_ORDER,
    _RENDERABLE_CONTEXT_KEYS,
    DERIVED_SOURCES,
    INFERRED_MARK,
    MEMORY_BLOCK_HEADER,
    MEMORY_INFERRED_HEADER,
    SOURCE_BEHAVIORAL,
    SOURCE_EXPLICIT,
    SOURCE_INFERRED,
    SOURCE_STATED,
    STATED_SOURCES,
    _order_index,
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
    # Оба имени «сказал сам» — библиотечное и бэкендовое — это «как раньше».
    assert build_memory_block(ctx, sources=dict.fromkeys(ctx, SOURCE_STATED)) == baseline
    assert build_memory_block(ctx, sources=dict.fromkeys(ctx, SOURCE_EXPLICIT)) == baseline
    # А вот ОТСУТСТВИЕ ключа — не значение: происхождение не сообщили, рендер
    # прежний. Именно на этом держится байт-в-байт совместимость, и это НЕ то же
    # самое, что присланное незнакомое значение (см. тест ниже).
    assert build_memory_block(ctx, sources={"diet_type": SOURCE_STATED}) == baseline
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


# ---------------------------------------------------------------------------
# Словарь происхождения — три репозитория, один перечень.
#
# До этих тестов выводом считалось РОВНО слово "inferred". Внутренний PATCH
# бэкенда принимает четыре значения (explicit|behavioral|conversational|
# transactional), ночная инференция ставит пятое ("inferred"), стирание —
# шестое ("erased"). Значит факт с "behavioral" уезжал в блок как прямая речь
# клиента. Ошибка несимметрична, поэтому правило перевёрнуто: цитата — только
# объявленное «сказал сам», всё прочее — вывод.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("origin", sorted(DERIVED_SOURCES))
def test_every_known_derived_value_is_marked_not_only_inferred(origin: str) -> None:
    """Каждое значение словаря, кроме «сказал сам», обязано читаться выводом.

    Тест по ВСЕМУ перечню, а не по одному `behavioral`: дефект был в том, что
    правило знало ровно одно слово из шести.
    """
    out = build_memory_block({"busy_days": ["tue"]}, sources={"busy_days": origin})
    assert MEMORY_INFERRED_HEADER in out, origin
    assert f"- {INFERRED_MARK} Избегает: вторник" in out, origin
    # И строка стоит НИЖЕ заголовка-правила, а не в списке сказанного выше него.
    assert out.index(MEMORY_INFERRED_HEADER) < out.index("Избегает"), origin


@pytest.mark.parametrize("origin", sorted(STATED_SOURCES))
def test_stated_values_still_render_as_the_clients_own_words(origin: str) -> None:
    """Положительная стража.

    Без неё правило «считать выводом всё» прошло бы и на полностью сломанной
    правке: тесты выше зелены, даже если цитат не осталось вовсе. `explicit` —
    имя, которым помечает бэкенд, и это ровно то, что клиент сказал сам.
    """
    out = build_memory_block({"busy_days": ["tue"]}, sources={"busy_days": origin})
    assert "- Избегает: вторник" in out, origin
    assert INFERRED_MARK not in out, origin
    assert MEMORY_INFERRED_HEADER not in out, origin


def test_unknown_origin_value_is_read_as_derived() -> None:
    """Незнакомое значение — вывод, а не «как раньше».

    Словарь ведут три репозитория; незнакомое значение здесь означает не
    «ничего», а «кто-то научился ставить то, о чём библиотека не знает».
    Безопасная сторона — умолчание.
    """
    out = build_memory_block({"busy_days": ["tue"]}, sources={"busy_days": "чтототакое"})
    assert f"- {INFERRED_MARK} Избегает: вторник" in out
    assert MEMORY_INFERRED_HEADER in out


def test_absent_key_is_not_an_unknown_value() -> None:
    """Граница, на которой держится совместимость.

    Присланное незнакомое значение = вывод. НЕ присланный ключ = происхождение
    не сообщили, рендер прежний. Схлопни их — и бэкенд, ещё не отдающий
    провенанс, превратит всю анкету в догадку при первом же деплое.
    """
    ctx = {"busy_days": ["tue"], "diet_type": "vegan"}
    out = build_memory_block(ctx, sources={"busy_days": SOURCE_BEHAVIORAL})
    assert f"- {INFERRED_MARK} Избегает: вторник" in out   # ключ прислан
    assert "- Диета: vegan" in out                          # ключа нет — как раньше
    assert f"{INFERRED_MARK} Диета" not in out


def test_the_backend_wire_map_can_be_passed_through_as_is() -> None:
    """Реальная форма входа: карта `data_sources` внутреннего GET.

    Бэкенд отдаёт ВСЕ зелёные поля, непомеченные — со значением "explicit".
    Нормализовать на стороне вызывающего больше не нужно: помечена ровно одна
    строка, у которой происхождение действительно не клиентское.
    """
    ctx = {
        "preferred_time_slots": ["evening"],
        "diet_type": "vegan",
        "busy_days": ["tue"],
        "min_rating_preference": 4.8,
    }
    wire = dict.fromkeys(ctx, SOURCE_EXPLICIT) | {"busy_days": SOURCE_BEHAVIORAL}
    out = build_memory_block(ctx, sources=wire)
    assert f"- {INFERRED_MARK} Избегает: вторник" in out
    assert out.count(INFERRED_MARK) == 1
    for stated_line in ("- Диета: vegan", "- Минимальный рейтинг мастера: 4.8"):
        assert stated_line in out


def test_derived_budget_is_marked_for_every_derived_value_too() -> None:
    """Склеенная из двух ключей строка подчиняется тому же перевёрнутому правилу."""
    ctx = {"price_range_min": Decimal("1000"), "price_range_max": Decimal("2500")}
    for origin in sorted(DERIVED_SOURCES):
        assert INFERRED_MARK in build_memory_block(
            ctx, sources={"price_range_min": origin}
        ), origin
    assert INFERRED_MARK not in build_memory_block(
        ctx, sources={"price_range_min": SOURCE_EXPLICIT, "price_range_max": SOURCE_EXPLICIT}
    )


def test_the_two_sides_of_the_vocabulary_do_not_overlap() -> None:
    """Структурная стража перечня: значение не может быть цитатой и выводом сразу."""
    assert frozenset() == STATED_SOURCES & DERIVED_SOURCES
    assert SOURCE_STATED in STATED_SOURCES
    assert SOURCE_EXPLICIT in STATED_SOURCES
    assert SOURCE_INFERRED in DERIVED_SOURCES


def test_the_whole_declared_vocabulary_is_classified() -> None:
    """Каждое значение, которое контур умеет ставить, имеет объявленную сторону.

    Перечень бэкенда зашит здесь намеренно: если внутренний PATCH заведёт
    пятое значение, а библиотеку не научат, тест этого НЕ поймает — правило
    и так отнесёт новое значение к выводам. Тест ловит обратное и худшее:
    молчаливую пропажу значения из перечня.
    """
    backend_patch_choices = {
        "explicit", "behavioral", "conversational", "transactional",
    }
    backend_stamps_elsewhere = {"inferred", "erased"}
    for value in backend_patch_choices | backend_stamps_elsewhere:
        assert value in (STATED_SOURCES | DERIVED_SOURCES), value


# ---------------------------------------------------------------------------
# Приоритет и усечение (DRF-1374) — полная анкета, а не короткая.
#
# Прежние тесты усечения были зелены ровно потому, что коротки: на трёх-четырёх
# полях лимит max_facts=8 не срабатывает вовсе, и дефект приоритета не виден.
# Клетка ниже — полная зелёная анкета `users.UserPersonalContext` (12 колонок),
# то есть тот вход, который приходит от реального клиента с заполненным
# профилем.
# ---------------------------------------------------------------------------

# Двенадцать полей зелёной анкеты. Ровно колонки Ayla `users.UserPersonalContext`.
FULL_GREEN_FORM: dict[str, object] = {
    "preferred_districts": ["Арбеково", "Центр"],
    "preferred_time_slots": ["evening"],
    "price_range_min": Decimal("1000"),
    "price_range_max": Decimal("2500"),
    "diet_type": "vegan",
    "skin_sensitivities": ["никель"],  # рендерер её не знает — см. тест ниже
    "prefers_flexible_cancellation": True,
    "workplace_district": "Западная поляна",
    "home_district": "Терновка",
    "favorite_masters": ["m-1"],
    "min_rating_preference": 4.8,
    "busy_days": ["mon", "tue"],
}


def _fact_lines(block: str) -> list[str]:
    return [ln for ln in block.splitlines() if ln.startswith("- ")]


def test_full_green_form_keeps_the_budget() -> None:
    """Бюджет обязан пережить усечение на ПОЛНОЙ анкете.

    До DRF-1374 ключей `price_range_min`/`price_range_max` не было
    в `_FIELD_ORDER` (там лежала строка `price_range`, которой во входном
    словаре не бывает), поэтому бюджет получал приоритет «неизвестный»,
    уезжал в хвост и на полной анкете срезался `max_facts=8` ВСЕГДА.
    """
    out = build_memory_block(FULL_GREEN_FORM)
    assert "Бюджет 1000–2500 ₽" in out


def test_full_green_form_truncates_by_declared_priority() -> None:
    """Порядок строк = объявленный `_FIELD_ORDER`, а не порядок словаря.

    И срезается ровно хвост приоритетов, а не то, что случайно оказалось
    без записи в таблице.
    """
    lines = _fact_lines(build_memory_block(FULL_GREEN_FORM))
    assert lines == [
        "- Любимые мастера: id=m-1",
        "- Обычно выбирает время: вечер (после 18:00)",
        "- Бюджет 1000–2500 ₽",
        "- Ищет рядом с работой (Западная поляна)",
        "- Ищет рядом с домом (Терновка)",
        "- Предпочитает районы: Арбеково, Центр",
        "- Избегает: понедельник, вторник",
        "- Диета: vegan",
    ]
    # Отрезаны два последних по приоритету — и только они.
    # Решение владельца 25.08 поменяло местами диету и минимальный рейтинг.
    # Ожидание правится намеренно, а не подгоняется: мест восемь, строк
    # десять, и вопрос «кто уедет» — продуктовый, а не технический. До
    # DRF-1374 диета доезжала случайно (бюджет молча выпадал из-за мёртвой
    # строки); починка вернула бюджет на объявленное место и вытеснила её.
    # Владелец выбрал диету: минимальный рейтинг — фильтр поиска, а не
    # память о человеке.
    assert "Минимальный рейтинг" not in "\n".join(lines)
    assert "гибкую отмену" not in "\n".join(lines)


def test_full_green_form_renders_everything_when_budget_allows() -> None:
    """Без лимита видны все десять строк в порядке таблицы.

    Строк десять, а разбираемых ключей одиннадцать: бюджет склеен из двух.
    """
    lines = _fact_lines(build_memory_block(FULL_GREEN_FORM, max_facts=99))
    assert len(lines) == len(_FIELD_ORDER) == 10
    assert lines[-2:] == [
        "- Минимальный рейтинг мастера: 4.8",
        "- Предпочитает гибкую отмену",
    ]


def test_every_renderable_key_has_a_declared_priority() -> None:
    """Структурный гейт: ключ без объявленного приоритета — ошибка сборки, не хвост.

    Это и есть противоядие от класса дефекта DRF-1374. Механизм сортировки
    отказывает беззвучно: любой ключ, которого нет в `_FIELD_ORDER`, молча
    получает приоритет «в самый хвост» и на полной анкете исчезает из промпта.
    Ронять на РАНТАЙМЕ нельзя (§2: чужой/красный ключ обязан игнорироваться,
    а не валить диалог), поэтому громкость переносится сюда: добавил ветку
    рендера — обязан объявить приоритет, иначе тест красный.
    """
    undeclared = sorted(k for k in _RENDERABLE_CONTEXT_KEYS if _order_index(k) == len(_FIELD_ORDER))
    assert undeclared == [], (
        f"Ключи умеют рендериться, но не имеют приоритета и молча уедут в хвост: {undeclared}. "
        "Добавь имя строки в _FIELD_ORDER (и в _CONTEXT_KEY_TO_FIELD, если имя ключа отличается)."
    )


def test_field_order_has_no_dead_entries() -> None:
    """Обратная сторона: приоритет, до которого не дотягивается ни один ключ.

    Строка `price_range` жила в таблице мёртвой — она объявляла намерение,
    которое код не исполнял, потому что такого ключа в контексте нет.
    """
    reachable = {_CONTEXT_KEY_TO_FIELD.get(k, k) for k in _RENDERABLE_CONTEXT_KEYS}
    dead = [name for name in _FIELD_ORDER if name not in reachable]
    assert dead == [], f"Приоритет объявлен, но недостижим ни одним ключом контекста: {dead}"


def test_profile_field_the_renderer_does_not_know_is_ignored() -> None:
    """`skin_sensitivities` есть в анкете, но рендерер её не разбирает.

    Это НЕ дефект приоритета: поле не имеет ветки рендера вовсе (зона за
    пределами зелёного surfacing), поэтому оно не должно ни попадать в блок,
    ни занимать место в бюджете фактов.
    """
    out = build_memory_block(FULL_GREEN_FORM, max_facts=99)
    assert "никель" not in out
    assert "skin_sensitivities" not in out
