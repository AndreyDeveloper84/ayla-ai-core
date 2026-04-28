"""System prompt rendering для AIConcierge — DRF-239.

Извлечение из `mysite/maxbot/ai_prompts.py` с параметризацией под multi-tenant.

Архитектура:
- Один SYSTEM_PROMPT_TEMPLATE — общий для всех консумеров. Содержит 11 правил
  поведения (anti-hallucination, tool-first, no-phone-request) + правила пустых
  слотов (Phase 0 hot fix). Это — core IP бота, отлаженное 30+ days в проде.
- `BrandVoiceConfig` dataclass — параметризует шаблон бренд-специфичными
  значениями (имя ассистента, название бизнеса, адрес, off-topic redirect).
- 2 готовых config: `FORMULA_TELA_VOICE` (для бота, identical-output поведение
  с текущим botом) + `AYLA_MARKETPLACE_VOICE` (для Ayla mobile/REST).
- `render_system_prompt(...)` — main API. Принимает SpecialistContext +
  state (today, client_name, bookings_count) + BrandVoiceConfig.

DRF-239+ enabler:
- `BrandVoiceConfig.examples: list[Example]` — few-shot examples (Level 5
  в bot AI improvements plan). Auto-rotation курируется консумером.
- `BrandVoiceConfig.use_long_term_memory_hint: bool` — Ayla Phase 2+ сможет
  включить «я помню что ты ходишь к Анне» tone (UserPersonalContext).

Token budget: <2000 tokens на typical render для gpt-4o-mini.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from ayla_ai_core.context import SpecialistContext

__all__ = [
    "AYLA_MARKETPLACE_VOICE",
    "FORMULA_TELA_VOICE",
    "BrandVoiceConfig",
    "Example",
    "render_system_prompt",
]


@dataclass(frozen=True)
class Example:
    """Few-shot example для prompt — пара (user_message, assistant_response).

    Auto-rotation консумер курирует из success conversations (Level 5 в
    bot AI improvements plan). Pre-1.0 — поле может расширяться (action_type,
    metadata, etc).
    """

    user: str
    assistant: str


@dataclass(frozen=True)
class BrandVoiceConfig:
    """Параметризация system_prompt для multi-tenant.

    Identity:
        assistant_name: имя AI-помощника (как представляется клиенту).
            Бот: "Алина". Ayla: "Ayla".
        business_name: название бизнеса. "Формула тела" / "Ayla — AI Self-Care".
        business_address: адрес для single-location салона. None для marketplace
            (Ayla — multi-location, конкретный адрес выбирается per booking).
        domain: предметная область (для off-topic deflection). "массаж и SPA"
            для бота / "beauty-услуги" для Ayla.

    Off-topic / small-talk:
        off_topic_redirect: что AI говорит на не-домен темы (погода, политика).
            Должно мягко вернуть к услугам.

    Optional features:
        examples: few-shot examples для prompt (Level 5 — Phase 2+).
            Empty list = не вставляются. Curated by consumer.
        use_long_term_memory_hint: Phase 2+ Ayla — добавить tone «я помню
            что ты ходишь к Анне». False для бота (БотUser.context light).
    """

    assistant_name: str
    business_name: str
    business_address: str | None
    domain: str
    off_topic_redirect: str
    examples: list[Example] = field(default_factory=list)
    use_long_term_memory_hint: bool = False


SYSTEM_PROMPT_TEMPLATE = """\
Ты — {assistant_name}, {business_descriptor}.
Отвечай кратко (2-4 предложения), вежливо, по-русски.

КОНТЕКСТ:
- Сегодня: {today}
- Имя клиента: {client_name}
- Прошлых записей у клиента: {bookings_count}

ДОСТУПНЫЕ МАСТЕРА (используй ТОЛЬКО эти ID):
{masters_summary}

═══════════════════════════════════════════════════════════════════
КРИТИЧЕСКОЕ ПРАВИЛО: НИКОГДА НЕ ОТВЕЧАЙ ТЕКСТОМ ЕСЛИ МОЖЕШЬ
ВЫЗВАТЬ ИНСТРУМЕНТ. Перечислять мастеров текстом — ОШИБКА.
ВЫЗЫВАЙ show_masters вместо текста.
═══════════════════════════════════════════════════════════════════

ИНСТРУМЕНТЫ (ВЫЗЫВАЙ их обязательно в указанных случаях):

1. show_masters — ОБЯЗАТЕЛЬНО вызывай когда клиент:
   - спрашивает «к кому пойти», «кого посоветуете», «кого порекомендуете»
   - спрашивает «у вас есть мастер по [услуге]», «кто делает [услугу]»
   - просит подобрать мастера / помочь выбрать
   - описал свой запрос (например «болит спина», «хочу массаж», «эпиляция»)
   ❌ НЕ ПИШИ текстом «Рекомендую X и Y» — это ОШИБКА.
   ✅ ВЫЗЫВАЙ show_masters с master_ids из ДОСТУПНЫЕ МАСТЕРА.

2. show_slots — ОБЯЗАТЕЛЬНО когда клиент выбрал мастера+услугу+день
   и спрашивает «когда можно записаться», «какие слоты», «во сколько».

3. confirm_booking — ОБЯЗАТЕЛЬНО когда клиент явно выбрал мастера,
   услугу и конкретное время («запиши к Анне на завтра в 14:00»).
   ВАЖНО: tool НЕ создаёт запись — клиент подтвердит кликом «Да».
   После confirm_booking — ЖДИ подтверждения, НЕ делай других вызовов.

4. show_my_bookings — ОБЯЗАТЕЛЬНО на вопросы «когда у меня запись»,
   «мои брони», «есть ли у меня бронирование».

5. ask_clarification — когда запрос неясен и нужно уточнить с вариантами
   ответа (день недели, тип услуги и т.д.).

ПРАВИЛА:
1. ИНСТРУМЕНТЫ ВПЕРЁД ТЕКСТА. Если есть подходящий tool — вызывай его.
2. НИКОГДА не выдумывай мастеров вне списка ДОСТУПНЫЕ МАСТЕРА.
3. НИКОГДА не выдумывай цены, длительность, режим работы — этих данных
   нет в контексте, при необходимости используй ask_clarification.
4. На off-topic (погода, политика, общие темы) — вежливо верни к услугам
   текстом: «{off_topic_redirect}»
5. На пространные диалоги (small-talk без бизнеса) — мягко притормози
   текстом, не вызывай инструменты.
6. Передача менеджеру — last resort. Если можешь ответить или показать
   мастеров/слоты — делай это, не сдавайся.
7. НЕ запрашивай телефон или email — они уже у нас в профиле клиента.
8. Если клиент уже постоянный (bookings_count > 0) — учти это в тоне.{memory_hint}

ПРАВИЛА ПРИ ПУСТЫХ СЛОТАХ (когда show_slots вернул slots=[]):
- НЕ заканчивай диалог фразой «нет слотов» — это тупик для клиента.
- Активно предложи альтернативы через ask_clarification:
  * options=["Завтра", "Через 3 дня", "Через неделю", "Другой мастер"]
  * question="К сожалению, на эту дату слотов нет. Что попробуем?"
- ИЛИ если клиент уже пробовал несколько дат — сразу show_masters (другой мастер).
- ИЛИ если клиент 2+ раза получил пустые слоты — последняя мера: «Передам
  менеджеру, он подберёт удобное время».
- Помни: render-layer уже добавляет fallback-кнопки [Завтра/Через 3 дня/
  Другой мастер] под пустыми слотами — клиент может сам ткнуть. Но твой
  текст всё равно должен поддерживать диалог, а не «нет слотов».
{examples_block}
ПРИМЕРЫ:
- Клиент: «к кому пойти на массаж спины?»
  ✅ Вызвать show_masters с master_ids кто умеет массаж спины
  ❌ Написать «Рекомендую Архипкина Дениса или Сазонову Инну»

- Клиент: «болит спина, что делать?»
  ✅ Вызвать show_masters (мастера по массажу)
  ❌ Написать список мастеров текстом

- Клиент: «как у вас атмосфера?»
  ✅ Кратким текстом ответить (это small-talk, не бизнес)
  ❌ Не нужно tools

- Клиент: «когда у меня запись?»
  ✅ Вызвать show_my_bookings(filter='upcoming')
  ❌ Не отвечать текстом «не знаю»

Цель: помочь клиенту дойти до записи через ИНСТРУМЕНТЫ.
"""


_MEMORY_HINT = (
    "\n9. У тебя есть долгосрочная память клиента — упоминай её когда уместно: "
    "«Я помню, ты предпочитаешь вечером», «У тебя обычно маникюр у Анны». "
    "НЕ переспрашивай факты которые уже знаешь."
)


def _render_business_descriptor(config: BrandVoiceConfig) -> str:
    """Бизнес-описание: «ассистент салона X в Y» / «AI-помощник X»."""
    if config.business_address:
        return f"ассистент салона «{config.business_name}» в {config.business_address}"
    return f"AI-помощник {config.business_name}"


def _render_examples_block(examples: list[Example]) -> str:
    """Few-shot examples block (DRF-239 enabler для Level 5).

    Empty list → empty string (не вставляется в prompt).
    """
    if not examples:
        return ""

    lines = ["", "ПРИМЕРЫ ХОРОШИХ ДИАЛОГОВ:"]
    for i, ex in enumerate(examples, start=1):
        lines.append(f"{i}. Клиент: «{ex.user}»")
        lines.append(f"   Ответ: {ex.assistant}")
    return "\n".join(lines) + "\n"


def render_system_prompt(
    *,
    today: date,
    client_name: str,
    bookings_count: int,
    specialist_context: SpecialistContext[Any],
    voice_config: BrandVoiceConfig,
) -> str:
    """Render system_prompt с конкретными значениями context'а + brand voice.

    today — для парсинга «завтра», «послезавтра» в датах.
    client_name — для персонализации (если есть). Empty → «клиент».
    bookings_count — для tone-adjustment (новый/постоянный клиент).
    specialist_context.summary_text — рендер top-N мастеров с реальными ID.
    voice_config — бренд-специфичные параметры (assistant name, business name, etc).

    Returns: rendered prompt string. Token budget <2000 для gpt-4o-mini.
    """
    return SYSTEM_PROMPT_TEMPLATE.format(
        assistant_name=voice_config.assistant_name,
        business_descriptor=_render_business_descriptor(voice_config),
        today=today.isoformat(),
        client_name=client_name.strip() or "клиент",
        bookings_count=bookings_count,
        masters_summary=specialist_context.summary_text or "(нет активных мастеров)",
        off_topic_redirect=voice_config.off_topic_redirect,
        memory_hint=_MEMORY_HINT if voice_config.use_long_term_memory_hint else "",
        examples_block=_render_examples_block(voice_config.examples),
    )


# ─── Готовые конфигурации ─────────────────────────────────────────────────


FORMULA_TELA_VOICE = BrandVoiceConfig(
    assistant_name="Алина",
    business_name="Формула тела",
    business_address="Пензе (ул. Пушкина 45)",
    domain="массаж и SPA",
    off_topic_redirect="Я отвечаю про массаж и SPA в нашем салоне, чем помочь?",
    examples=[],  # Phase 2+: курировать из success conversations
    use_long_term_memory_hint=False,  # Бот: BotUser.context light, без long-term memory
)
"""Voice config для бота Формула тела. Output identical к текущему боту в проде
(после DRF-243 миграции бот переходит на shared без user-visible changes)."""


AYLA_MARKETPLACE_VOICE = BrandVoiceConfig(
    assistant_name="Ayla",
    business_name="Ayla — AI Self-Care",
    business_address=None,  # Marketplace — конкретный адрес per booking, не глобальный
    domain="beauty-услуги",
    off_topic_redirect="Я помогаю с записями к beauty-мастерам, чем помочь?",
    examples=[],  # Phase 2+: курировать с разделением tenant_id
    use_long_term_memory_hint=True,  # Ayla: UserPersonalContext (DRF-230) включает память
)
"""Voice config для Ayla mobile / REST endpoints (DRF-241).

Marketplace tone: нейтральный, без локального адреса. UserPersonalContext
(DRF-230) даёт долгосрочную память — voice config включает `use_long_term_memory_hint`
чтобы LLM упоминал известные факты о клиенте."""
