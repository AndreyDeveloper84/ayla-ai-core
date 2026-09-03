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

* значение из :data:`STATED_SOURCES` — человек сказал это сам. Рендерится
  как раньше, без метки.
* ЛЮБОЕ другое присланное значение — вывод системы. Факт уходит в отдельную
  секцию под :data:`MEMORY_INFERRED_HEADER` и получает префикс
  :data:`INFERRED_MARK`, чтобы граница пережила усечение и перестановку.
* отсутствие ключа — не значение, а «происхождение не сообщили»: рендер
  как раньше.

Почему список закрыт со стороны цитаты, а не со стороны вывода. Словарь
происхождения ведут три репозитория, и они его уже разошлись: библиотека
завела `stated`, бэкенд ещё раньше писал `explicit` и открыл своему
внутреннему PATCH четыре значения (`explicit|behavioral|conversational|
transactional`), плюс ночная инференция ставит `inferred`, а стирание —
`erased`. Сравнение на точное совпадение с одним «inferred» пропускало бы
все остальные выводы в блок как прямую речь клиента. Ошибка здесь
несимметрична: пометить догадкой сказанное — неприятно, выдать домысел за
слова клиента — то, чего продукт избегает намеренно. Поэтому цитатой
считается только объявленное «сказал сам», а незнакомое значение —
выводом: безопасная сторона стоит умолчанием, и появление в контуре нового
вида вывода не требует правки библиотеки.

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
    "DERIVED_SOURCES",
    "INFERRED_MARK",
    "MEMORY_BLOCK_HEADER",
    "MEMORY_INFERRED_HEADER",
    "SOURCE_BEHAVIORAL",
    "SOURCE_CONVERSATIONAL",
    "SOURCE_ERASED",
    "SOURCE_EXPLICIT",
    "SOURCE_INFERRED",
    "SOURCE_STATED",
    "SOURCE_TRANSACTIONAL",
    "STATED_SOURCES",
    "build_memory_block",
]

MEMORY_BLOCK_HEADER = "[ПАМЯТЬ О ПОЛЬЗОВАТЕЛЕ — использовать как помощь, объяснять почему]"

# Словарь происхождения факта. Граница ровно одна — «сказал сам» против
# «вывели», шкалы доверия здесь нет. Но ИМЁН у этой границы в контуре больше
# одного: библиотека завела своё («stated»), бэкенд писал своё («explicit»)
# ещё до неё, и три репозитория независимо доучивали список. Поэтому перечень
# живёт здесь — в единственном месте, которое читают оба потребителя.

# --- сторона «человек сказал сам» ---
SOURCE_STATED = "stated"      # имя библиотеки
SOURCE_EXPLICIT = "explicit"  # имя бэкенда для ровно того же самого
                              # (users.UserPersonalContext.data_sources,
                              #  _SOURCE_CHOICES внутреннего PATCH)

# --- сторона «это не слова клиента» ---
SOURCE_INFERRED = "inferred"            # ночная инференция по истории броней
# behavioral / conversational / transactional — три значения, которые принимает
# _SOURCE_CHOICES внутреннего PATCH бэкенда.
SOURCE_BEHAVIORAL = "behavioral"
SOURCE_CONVERSATIONAL = "conversational"
SOURCE_TRANSACTIONAL = "transactional"
SOURCE_ERASED = "erased"                # надгробие стёртого поля

# Закрытый список: цитатой считается ТОЛЬКО значение отсюда. Список
# load-bearing — именно он, а не перечень выводов, решает исход. Пополнять
# его можно лишь осознанно: каждое добавленное значение — это разрешение
# выдать факт за прямую речь клиента.
STATED_SOURCES: frozenset[str] = frozenset({SOURCE_STATED, SOURCE_EXPLICIT})

# Открытый список: известные контуру значения, которые цитатой НЕ являются.
# На исход не влияет (правило — дополнение к STATED_SOURCES, а не членство
# здесь), поэтому появление в контуре нового вывода не требует правки
# библиотеки. Держим его, чтобы у трёх репозиториев был один перечень, на
# который можно сослаться, и чтобы тесты гоняли рендер по всему словарю.
# `erased` — строго говоря не вывод, а надгробие; общее у них то, что
# ни то, ни другое нельзя подать как слова клиента.
DERIVED_SOURCES: frozenset[str] = frozenset(
    {
        SOURCE_INFERRED,
        SOURCE_BEHAVIORAL,
        SOURCE_CONVERSATIONAL,
        SOURCE_TRANSACTIONAL,
        SOURCE_ERASED,
    }
)

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

# Приоритет СТРОК блока (для top-N усечения при лимите). Имена здесь — имена
# рендерящихся строк, а не обязательно ключей входного словаря: строка
# «Бюджет» склеивается из двух ключей и живёт под собственным именем.
_FIELD_ORDER = (
    "favorite_masters",
    "preferred_time_slots",
    "price_range",
    "workplace_district",
    "home_district",
    "preferred_districts",
    "busy_days",
    # Решение владельца 25.08: диета выше минимального рейтинга. Мест в блоке
    # восемь, а строк десять, и до DRF-1374 диета доезжала только потому, что
    # бюджет молча выпадал из-за мёртвой строки. Починка вернула бюджет на
    # объявленное третье место и вытеснила диету за черту. Владелец выбрал
    # поднять её: минимальный рейтинг — фильтр поиска, а не память о человеке,
    # диета же говорит о собеседнике много. Недельная картина питания едет
    # отдельным блоком (DRF-1284) и от этого порядка не зависит.
    "diet_type",
    "min_rating_preference",
    "prefers_flexible_cancellation",
)

# Ключ контекста -> имя строки в _FIELD_ORDER, когда они НЕ совпадают.
# Единственный такой случай — бюджет: во входном словаре его нет вовсе,
# есть price_range_min/price_range_max. Без этой карты оба ключа получали
# приоритет «неизвестный» и уезжали в хвост, а строка "price_range"
# в _FIELD_ORDER была мёртвой (DRF-1374).
_CONTEXT_KEY_TO_FIELD = {
    "price_range_min": "price_range",
    "price_range_max": "price_range",
}

# Ключи контекста, которые рендерер умеет разбирать. Всё остальное молча
# игнорируется (§2: не выдумываем; чужой/красный ключ не должен ронять
# диалог). Множество load-bearing: это гейт цикла, поэтому ветка без записи
# здесь — мёртвая ветка, а запись без ветки — мёртвая запись. Тест
# test_every_renderable_key_has_a_declared_priority требует, чтобы у каждого
# ключа отсюда был объявленный приоритет: молчаливый уезд в хвост больше
# невозможен.
_RENDERABLE_CONTEXT_KEYS = frozenset(
    {
        "favorite_masters",
        "preferred_time_slots",
        "price_range_min",
        "price_range_max",
        "workplace_district",
        "home_district",
        "preferred_districts",
        "busy_days",
        "min_rating_preference",
        "diet_type",
        "prefers_flexible_cancellation",
    }
)

# Приоритет «в самый хвост» — для ключей без объявленного порядка.
_ORDER_UNDECLARED = len(_FIELD_ORDER)


def _order_index(key: str) -> int:
    """Приоритет ключа контекста по _FIELD_ORDER (хвост, если не объявлен)."""
    field = _CONTEXT_KEY_TO_FIELD.get(key, key)
    return _FIELD_ORDER.index(field) if field in _FIELD_ORDER else _ORDER_UNDECLARED


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
        sources: по-полю происхождение факта. Значение из
            :data:`STATED_SOURCES` (``"stated"`` — имя библиотеки,
            ``"explicit"`` — имя бэкенда) = человек сказал сам, рендерится
            как раньше. ЛЮБОЕ другое значение — включая незнакомое
            библиотеке — считается выводом: поле уходит в отдельную секцию
            под :data:`MEMORY_INFERRED_HEADER` с префиксом
            :data:`INFERRED_MARK`. Отсутствие ключа значением не является —
            это «происхождение не сообщили», рендер как раньше, поэтому БЕЗ
            этого аргумента вывод байт-в-байт совпадает с прежним. Карту
            бэкенда (``data_sources`` внутреннего GET) можно передавать сюда
            как есть, без предварительной нормализации.
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

    def _emit(field: str, text: str, *origin_fields: str) -> None:
        c = float(conf.get(field, 1.0))
        # `origin_fields` exists for the one line built from SEVERAL context
        # keys: бюджет склеивается из price_range_min/max, а рендерится под
        # именем `price_range`, которого во входном словаре нет вовсе. Без
        # этого выведенный бюджет молча остался бы непомеченным — тихая дыра
        # ровно того сорта, который этот параметр и закрывает.
        # Одна строка — одно происхождение: смесь цитаты с догадкой честнее
        # пометить догадкой.
        #
        # Цитатой считается ТОЛЬКО значение из закрытого STATED_SOURCES. Любое
        # другое присланное значение — вывод, в том числе незнакомое: словарь
        # ведут три репозитория, и незнакомое значение здесь означает не
        # «ничего», а «кто-то из них научился ставить то, о чём библиотека не
        # знает». Ошибка несимметрична — пометить сказанное догадкой неприятно,
        # выдать догадку за слова клиента продукт избегает намеренно, — поэтому
        # умолчанием стоит безопасная сторона.
        #
        # ОТСУТСТВИЕ ключа — не значение: происхождение просто не сообщили, и
        # строка рендерится как раньше. На этом держится байт-в-байт совместимость
        # (без `sources` в словаре нет ни одного ключа) и деплой врозь: бэкенд,
        # ещё не приславший провенанс, не превращает всю анкету в догадку.
        inferred = any(
            src[f] not in STATED_SOURCES for f in (field, *origin_fields) if f in src
        )
        if c < CONF_SOFT:
            # «Стоит уточнить» — это уже вопрос, а не утверждение: выдать его
            # за слова клиента нельзя, метка была бы лишним шумом.
            to_clarify.append(text)
        else:
            facts.append((_soften(text, c), inferred))

    ordered = sorted(context.keys(), key=_order_index)
    # price склеиваем из min/max — обрабатываем один раз.
    price_done = False

    for field in ordered:
        # Неизвестные ключи молча игнорируем (§2: не выдумываем). Гейт стоит
        # ДО веток, чтобы множество известных ключей было единственным —
        # ветка, не объявленная в нём, просто не выполнится, и это заметит тест.
        if field not in _RENDERABLE_CONTEXT_KEYS:
            continue
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
            _emit("price_range", text, "price_range_min", "price_range_max")
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
