"""OpenAI tool definitions для AIConcierge — generic over ID type (DRF-238).

Эти JSON-Schema передаются в `chat.completions.create(tools=[...])` — LLM
эмиттит tool_call, который `tool_handlers.dispatch_tool_call` валидирует и
формирует action_data.

5 tools в MVP. ActionType — стабильный wire-format между LLM, persisted
Message.action_type, и UI render layer консумера.

ID type:
- Bot Формулы (Master.id = int): `build_tool_definitions("integer")` —
  default, доступно как константа TOOL_DEFINITIONS для backward compat.
- Ayla (SpecialistProfile.id = UUID): `build_tool_definitions("string")`
  — UUID передаётся как строка в JSON.

Wire field names — `master_id` / `service_id` остались неизменными после
DRF-238 (LLM trained behavior + бот в проде уже эмиттит эти имена).
"""
from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Literal

__all__ = [
    "ASK_CLARIFICATION",
    "CONFIRM_BOOKING",
    "SHOW_MASTERS",
    "SHOW_MY_BOOKINGS",
    "SHOW_SLOTS",
    "TOOL_DEFINITIONS",
    "ActionType",
    "build_tool_definitions",
]


# Тип ID в JSON Schema. "integer" для бота (int), "string" для Ayla (UUID).
IdSchemaType = Literal["integer", "string"]


def _show_masters(id_type: IdSchemaType) -> dict[str, Any]:
    return {
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
                        "items": {"type": id_type},
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


def _show_slots(id_type: IdSchemaType) -> dict[str, Any]:
    return {
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
                    "master_id": {"type": id_type, "description": "ID мастера из контекста"},
                    "service_id": {"type": id_type, "description": "ID услуги из контекста"},
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


def _confirm_booking(id_type: IdSchemaType) -> dict[str, Any]:
    return {
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
                    "master_id": {"type": id_type},
                    "service_id": {"type": id_type},
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


def _show_my_bookings() -> dict[str, Any]:
    return {
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


def _ask_clarification() -> dict[str, Any]:
    return {
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


def build_tool_definitions(id_schema_type: IdSchemaType = "integer") -> list[dict[str, Any]]:
    """Factory: 5 tool definitions с указанным JSON Schema id type.

    - "integer" — default, для int IDs (бот Формулы, Master.id).
    - "string" — для UUID consumers (Ayla, SpecialistProfile.id передаётся
      как str(uuid)).

    Возвращает свежий list каждый вызов (mutable dicts — caller может
    модифицировать без эффекта на других consumers).
    """
    return [
        _show_masters(id_schema_type),
        _show_slots(id_schema_type),
        _confirm_booking(id_schema_type),
        _show_my_bookings(),
        _ask_clarification(),
    ]


# ─── Backward compat constants (default int IDs для бота) ─────────────────

# Default tool definitions — int IDs. Бот / legacy callers могут импортировать
# напрямую `from ayla_ai_core import TOOL_DEFINITIONS`. Ayla / multi-tenant
# консумеры используют `build_tool_definitions("string")`.
#
# v0.7.2 (DRF-679): module-level constant — read-only. Wrapped via
# `MappingProxyType` + outer `tuple` so consumer mutation
# (`TOOL_DEFINITIONS[0]["x"] = "y"` or `TOOL_DEFINITIONS.append(...)`) raises
# at the top level. Callers that need a mutable copy use
# `build_tool_definitions(...)` — it returns a fresh `list[dict]` per call.
#
# Note: shallow immutability only — `TOOL_DEFINITIONS[0]["function"]["x"] = …`
# still works because nested dicts are not wrapped. Deeper immutability
# would change the signature of consumer code that legitimately reads
# `tool["function"]["parameters"]` — defer to a future minor.
TOOL_DEFINITIONS: tuple[Mapping[str, Any], ...] = tuple(
    MappingProxyType(d) for d in build_tool_definitions("integer")
)

SHOW_MASTERS: Mapping[str, Any] = TOOL_DEFINITIONS[0]
"""DEPRECATED: используй build_tool_definitions(...). Alias для backward compat."""

SHOW_SLOTS: Mapping[str, Any] = TOOL_DEFINITIONS[1]
"""DEPRECATED: используй build_tool_definitions(...). Alias для backward compat."""

CONFIRM_BOOKING: Mapping[str, Any] = TOOL_DEFINITIONS[2]
"""DEPRECATED: используй build_tool_definitions(...). Alias для backward compat."""

SHOW_MY_BOOKINGS: Mapping[str, Any] = TOOL_DEFINITIONS[3]
"""DEPRECATED: используй build_tool_definitions(...). Alias для backward compat."""

ASK_CLARIFICATION: Mapping[str, Any] = TOOL_DEFINITIONS[4]
"""DEPRECATED: используй build_tool_definitions(...). Alias для backward compat."""


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
