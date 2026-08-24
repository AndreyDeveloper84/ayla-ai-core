"""Memory-block builder — персональная память в system_prompt concierge.

Контракт: ai-bot-platform/docs/plans/2026-07-03-AI_CONCIERGE_MEMORY_PROMPT_SPEC.md
(§1 что входит, §2 запреты, §3 surfacing, §4 структура).

Принцип: память = помощь, не слежка. В промпт кладём ТОЛЬКО зелёную зону,
с confidence-aware формулировкой (уверенный факт утверждаем, слабый — помечаем
«уточнить»). НЕ включаем red/yellow, сырые логи, не выдумываем id.

Вызывающая сторона (бот) сама решает — инъектить блок или нет (consent-гейт
memory_green проверяется ДО этого модуля, см. MEMORY_CONSENT_SPEC).

Происхождение факта (P0-3, OD_C04_GROUNDED_WHY §1)
--------------------------------------------------
Модель обязана отличать «человек сказал сам» от «мы вывели из истории».
До этого параметра библиотека такой границы не знала: она умела только
`confidence` — параметр ОТОБРАЖЕНИЯ («кажется, …»), не происхождения.
Догадка с confidence=1.0 приходила в промпт неотличимой от прямой цитаты.

Поэтому `build_memory_block` принимает `sources: {field: origin}`:

* ``SOURCE_INFERRED`` («inferred») — вывод системы. Факт уходит в отдельную
  секцию под :data:`MEMORY_INFERRED_HEADER` и получает префикс
  :data:`INFERRED_MARK`, чтобы граница пережила усечение и перестановку.
* всё остальное (в т.ч. отсутствие ключа) — рендерится как раньше.

Обратная совместимость жёсткая: без `sources` вывод БАЙТ-В-БАЙТ прежний.
`confidence` и `source` — ортогональны и НЕ выводятся друг из друга:
«насколько уверены» и «кто это сказал» — разные вопросы.
"""
from __future__ import annotations

from decimal import Decimal
from typing import Any

__all__ = [
    "CONF_ASSERT",
    "CONF_SOFT",
    "INFERRED_MARK",
    "MEMORY_BLOCK_HEADER",
    "MEMORY_INFERRED_HEADER",
    "SOURCE_INFERRED",
    "SOURCE_STATED",
    "build_memory_block",
]

MEMORY_BLOCK_HEADER = "[ПАМЯТЬ О ПОЛЬЗОВАТЕЛЕ — использовать как помощь, объяснять почему]"

# Происхождение факта — ровно ДВА значения, шкалы доверия здесь нет.
SOURCE_STATED = "stated"      # человек сказал это сам
SOURCE_INFERRED = "inferred"  # система вывела из истории/поведения

# Заголовок секции выведенных фактов. Несёт ПРАВИЛО (OD_C04 §1: WHY —
# только пересказ сказанного), а не только метку: без правила модель
# знает границу, но не знает, что с ней делать.
MEMORY_INFERRED_HEADER = (
    "Это не слова клиента, а наша догадка из истории. "
    "Не ссылаться на неё как на сказанное клиентом:"
)
# Построчная метка: секция может быть усечена или переставлена, метка — нет.
INFERRED_MARK = "(вывод)"

# Пороги уверенности (PROMPT_SPEC §8.1): >=0.8 утверждать, <0.4 уточнять.
CONF_ASSERT = 0.8
CONF_SOFT = 0.4

# Ключ TimeSlot -> человекочитаемо (green).
_TIME_SLOT_LABELS = {
    "morning": "утро",
    "day": "день",
    "evening": "вечер (после 18:00)",
    "night": "поздний вечер",
}
# ISO weekday short -> русское.
_WEEKDAY_LABELS = {
    "mon": "понедельник",
    "tue": "вторник",
    "wed": "среду",
    "thu": "четверг",
    "fri": "пятницу",
    "sat": "субботу",
    "sun": "воскресенье",
}

# Приоритет полей в блоке (для top-N усечения при лимите).
_FIELD_ORDER = (
    "favorite_masters",
    "preferred_time_slots",
    "price_range",
    "workplace_district",
    "home_district",
    "preferred_districts",
    "busy_days",
    "min_rating_preference",
    "diet_type",
    "prefers_flexible_cancellation",
)


def _soften(text: str, conf: float) -> str:
    """Обернуть утверждение по уверенности: >=0.8 как есть, 0.4-0.8 мягко."""
    if conf >= CONF_ASSERT:
        return text
    return f"кажется, {text}"


def build_memory_block(
    context: dict[str, Any],
    *,
    confidences: dict[str, float] | None = None,
    sources: dict[str, str] | None = None,
    master_names: dict[str, str] | None = None,
    max_facts: int = 8,
) -> str:
    """Собрать memory-block для system_prompt из зелёной памяти.

    Args:
        context: зелёные поля памяти (значения). Пустые/None пропускаются.
            Ожидаемые ключи: favorite_masters, preferred_time_slots,
            price_range_min, price_range_max, workplace_district,
            home_district, preferred_districts, busy_days,
            min_rating_preference, diet_type, prefers_flexible_cancellation.
        confidences: по-полю 0..1 (default 1.0 = explicit). <0.4 -> в «уточнить».
            Это параметр ОТОБРАЖЕНИЯ («кажется, …»), НЕ происхождения — см.
            `sources`.
        sources: по-полю происхождение факта: :data:`SOURCE_STATED` (человек
            сказал сам) или :data:`SOURCE_INFERRED` (вывела система). Поля
            с ``inferred`` уходят в отдельную секцию под
            :data:`MEMORY_INFERRED_HEADER` с префиксом :data:`INFERRED_MARK`.
            Отсутствие ключа / неизвестное значение = рендер как раньше, поэтому
            БЕЗ этого аргумента вывод байт-в-байт совпадает с прежним.
        master_names: uuid -> имя мастера для surfacing (id всё равно даётся LLM).
        max_facts: верхний лимит фактов в блоке (top-N по _FIELD_ORDER).

    Returns:
        Markdown-блок для промпта, либо "" если памяти нет (caller ничего не инъектит).

    Запреты (§2): red/yellow зоны, сырые логи, выдуманные id сюда не попадают —
    на вход подаётся только зелёный словарь значений.
    """
    conf = confidences or {}
    src = sources or {}
    names = master_names or {}

    # (текст, выведен ли факт) — происхождение едет РЯДОМ с фактом до самого
    # рендера. Схлопни его в строку раньше — и оно потеряется ровно так же,
    # как терялось до P0-3.
    facts: list[tuple[str, bool]] = []   # утверждаемые (assert/soft)
    to_clarify: list[str] = []           # низкая уверенность -> уточнить вопросом

    def _emit(field: str, text: str) -> None:
        c = float(conf.get(field, 1.0))
        inferred = src.get(field) == SOURCE_INFERRED
        if c < CONF_SOFT:
            # «Стоит уточнить» — это уже вопрос, а не утверждение: выдать его
            # за слова клиента нельзя, метка была бы лишним шумом.
            to_clarify.append(text)
        else:
            facts.append((_soften(text, c), inferred))

    ordered = sorted(
        context.keys(),
        key=lambda k: _FIELD_ORDER.index(k) if k in _FIELD_ORDER else len(_FIELD_ORDER),
    )
    # price склеиваем из min/max — обрабатываем один раз.
    price_done = False

    for field in ordered:
        value = context.get(field)
        if not value:  # None / "" / [] / {} / 0 / False -> нет факта (сужает None для mypy)
            continue

        if field == "favorite_masters":
            rendered = ", ".join(
                f"{names[str(mid)]} (id={mid})" if str(mid) in names else f"id={mid}"
                for mid in value
            )
            _emit("favorite_masters", f"Любимые мастера: {rendered}")
        elif field == "preferred_time_slots":
            labels = [_TIME_SLOT_LABELS.get(s, s) for s in value]
            _emit("preferred_time_slots", f"Обычно выбирает время: {', '.join(labels)}")
        elif field in ("price_range_min", "price_range_max"):
            if price_done:
                continue
            lo = context.get("price_range_min")
            hi = context.get("price_range_max")
            if lo in (None, "") and hi in (None, ""):
                continue
            price_done = True
            if lo not in (None, "") and hi not in (None, ""):
                text = f"Бюджет {_num(lo)}–{_num(hi)} ₽"
            elif hi not in (None, ""):
                text = f"Бюджет до {_num(hi)} ₽"
            else:
                text = f"Бюджет от {_num(lo)} ₽"
            _emit("price_range", text)
        elif field == "workplace_district":
            _emit("workplace_district", f"Ищет рядом с работой ({value})")
        elif field == "home_district":
            _emit("home_district", f"Ищет рядом с домом ({value})")
        elif field == "preferred_districts":
            _emit("preferred_districts", f"Предпочитает районы: {', '.join(value)}")
        elif field == "busy_days":
            days = [_WEEKDAY_LABELS.get(d, d) for d in value]
            _emit("busy_days", f"Избегает: {', '.join(days)}")
        elif field == "min_rating_preference":
            _emit("min_rating_preference", f"Минимальный рейтинг мастера: {value}")
        elif field == "diet_type":
            _emit("diet_type", f"Диета: {value}")
        elif field == "prefers_flexible_cancellation" and value:
            _emit("prefers_flexible_cancellation", "Предпочитает гибкую отмену")
        # неизвестные ключи молча игнорируем (§2: не выдумываем)

    if not facts and not to_clarify:
        return ""

    # Лимит держим на ОБЩЕМ списке в порядке _FIELD_ORDER, а не на каждой
    # группе: иначе пометка происхождения молча удваивала бы бюджет блока.
    facts = facts[:max_facts]
    stated = [text for text, inferred in facts if not inferred]
    derived = [text for text, inferred in facts if inferred]

    lines = [MEMORY_BLOCK_HEADER]
    lines.extend(f"- {text}" for text in stated)
    if derived:
        lines.append(MEMORY_INFERRED_HEADER)
        lines.extend(f"- {INFERRED_MARK} {text}" for text in derived)
    if to_clarify:
        lines.append(f"Стоит уточнить (низкая уверенность): {'; '.join(to_clarify)}")
    return "\n".join(lines)


def _num(v: Any) -> str:
    """Целочисленное представление денег (Decimal/float/int) без хвоста .0."""
    if isinstance(v, Decimal) or (isinstance(v, float) and v.is_integer()):
        v = int(v)
    return str(v)
