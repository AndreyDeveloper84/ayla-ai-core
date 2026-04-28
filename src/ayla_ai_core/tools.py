"""OpenAI tool definitions для AIConcierge.

Адаптация из `mysite/maxbot/ai_tools.py`. Эти JSON-Schema передаются в
`chat.completions.create(tools=[...])` — LLM эмиттит tool_call, который
`tool_handlers.dispatch_tool_call` валидирует и формирует action_data.

5 tools в MVP. ActionType — стабильный wire-format между LLM, persisted
Message.action_type, и UI render layer консумера.

В DRF-238 master_id/service_id (int) станут generic (int для бота, UUID
для Ayla); сейчас остаются int как в исходном коде Формулы.
"""
from __future__ import annotations

__all__ = [
    "ASK_CLARIFICATION",
    "CONFIRM_BOOKING",
    "SHOW_MASTERS",
    "SHOW_MY_BOOKINGS",
    "SHOW_SLOTS",
    "TOOL_DEFINITIONS",
    "ActionType",
]


SHOW_MASTERS = {
    "type": "function",
    "function": {
        "name": "show_masters",
        "description": (
            "Показать рекомендованных мастеров клиенту с обоснованием выбора. "
            "Используй после того как клиент выразил пожелание (тип услуги, "
            "ожидания) и в контексте промпта есть подходящие кандидаты. "
            "НЕ выдумывай master_id — используй ТОЛЬКО ID из контекста."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "master_ids": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "ID мастеров из контекста, отсортированные по релевантности",
                    "maxItems": 5,
                },
                "match_scores": {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 0, "maximum": 100},
                    "description": "Оценка совпадения 0-100 для каждого, в том же порядке",
                },
                "match_reasons": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "description": "Краткие причины (1-3 на мастера) почему подходит",
                },
                "explanation": {
                    "type": "string",
                    "description": "Общее объяснение почему именно эти мастера выбраны",
                },
            },
            "required": ["master_ids", "explanation"],
        },
    },
}


SHOW_SLOTS = {
    "type": "function",
    "function": {
        "name": "show_slots",
        "description": (
            "Показать свободные слоты для конкретного мастера + услуги на дату. "
            "Используй когда клиент выбрал и мастера, и услугу, и день."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "master_id": {"type": "integer", "description": "ID мастера из контекста"},
                "service_id": {"type": "integer", "description": "ID услуги из контекста"},
                "date": {
                    "type": "string",
                    "format": "date",
                    "description": "Целевая дата YYYY-MM-DD",
                },
            },
            "required": ["master_id", "service_id", "date"],
        },
    },
}


CONFIRM_BOOKING = {
    "type": "function",
    "function": {
        "name": "confirm_booking",
        "description": (
            "Показать карточку подтверждения записи. ВАЖНО: эта функция "
            "не создаёт запись — клиент подтверждает кликом на кнопку «Да», "
            "и только тогда запись уйдёт в систему. Эмитти ТОЛЬКО когда "
            "клиент явно выбрал мастера + услугу + конкретное время."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "master_id": {"type": "integer"},
                "service_id": {"type": "integer"},
                "datetime": {
                    "type": "string",
                    "format": "date-time",
                    "description": "Начало слота в ISO 8601 с timezone, e.g. 2026-04-28T14:00:00+03:00",
                },
            },
            "required": ["master_id", "service_id", "datetime"],
        },
    },
}


SHOW_MY_BOOKINGS = {
    "type": "function",
    "function": {
        "name": "show_my_bookings",
        "description": (
            "Показать существующие записи клиента. Используй когда спрашивает "
            "«когда у меня запись», «есть ли у меня бронь», «мои записи»."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "filter": {
                    "type": "string",
                    "enum": ["upcoming", "past", "all"],
                    "description": "Какой набор записей показать",
                },
            },
        },
    },
}


ASK_CLARIFICATION = {
    "type": "function",
    "function": {
        "name": "ask_clarification",
        "description": (
            "Задать уточняющий вопрос с предложенными вариантами ответа. "
            "Используй когда запрос неясен (например «какой день удобнее?»)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "options": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Опциональные предзаполненные варианты ответа",
                    "maxItems": 5,
                },
            },
            "required": ["question"],
        },
    },
}


TOOL_DEFINITIONS = [
    SHOW_MASTERS,
    SHOW_SLOTS,
    CONFIRM_BOOKING,
    SHOW_MY_BOOKINGS,
    ASK_CLARIFICATION,
]


class ActionType:
    """Action types — стабильный wire-format. Имена совпадают с function.name.

    Используется в Message.action_type (persisted в БД консумера) и в
    UI render layer — какую карточку показывать пользователю.
    """

    SHOW_MASTERS = "show_masters"
    SHOW_SLOTS = "show_slots"
    CONFIRM_BOOKING = "confirm_booking"
    SHOW_MY_BOOKINGS = "show_my_bookings"
    ASK_CLARIFICATION = "ask_clarification"

    ALL_MVP = frozenset({
        SHOW_MASTERS,
        SHOW_SLOTS,
        CONFIRM_BOOKING,
        SHOW_MY_BOOKINGS,
        ASK_CLARIFICATION,
    })
