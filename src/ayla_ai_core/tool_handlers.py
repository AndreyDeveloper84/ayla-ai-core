"""Tool-call handlers — валидация LLM args + формирование action_data.

Адаптация из `mysite/maxbot/ai_tool_handlers.py`. Handlers SIDE-EFFECT-FREE:
- Не делают network calls (real fetch слотов / создание записей — в render layer
  и action_service консумера)
- Не пишут в БД (persistence — caller-side)
- Только validate args + формируют action_data для UI рендера

Anti-hallucination — критическая часть:
Каждый handler фильтрует ID через `context.candidate_ids` /
`context.candidate_service_ids` (frozenset из реальной БД). Если LLM выдумал
ID — `_fallback_clarification` возвращает ask_clarification вместо raise.
Клиент НИКОГДА не видит сломанную карточку.

Real-world incident 2026-04-27 (бот Формулы тела): gpt-4o-mini выдал
service_id=1 которого не было в БД, handler сфолбекнулся на ask_clarification —
клиент молча получил уточняющий вопрос, не error.

Для `handle_confirm_booking` — нужны имена мастера/услуги для UI карточки.
В боте они подтягиваются ORM-запросом. В shared package принимаем optional
`master_resolver` / `service_resolver` callables; если не предоставлены —
action_data возвращается без enrichment, caller дополняет в render layer.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from ayla_ai_core.context import MasterContext
from ayla_ai_core.tools import ActionType

__all__ = [
    "MasterResolver",
    "ServiceResolver",
    "ToolResult",
    "dispatch_tool_call",
    "handle_ask_clarification",
    "handle_confirm_booking",
    "handle_show_masters",
    "handle_show_my_bookings",
    "handle_show_slots",
]


logger = logging.getLogger("ayla_ai_core.tool_handlers")


# Optional resolvers для confirm_booking enrichment.
# Возвращают dict с полями name (str), price_from (Decimal | str | None),
# duration_min (int | None) — ИЛИ None если объект не найден.
MasterResolver = Callable[[int], dict[str, Any] | None]
ServiceResolver = Callable[[int], dict[str, Any] | None]


@dataclass(frozen=True)
class ToolResult:
    """Результат tool_handler — нормализованная пара (action_type, action_data).

    AIConcierge сохранит как Message.action_type / Message.action_data в
    persistence слое консумера.
    """

    action_type: str
    action_data: dict[str, Any]


# ─── Helpers ──────────────────────────────────────────────────────────────


def _safe_int(value: Any) -> int | None:
    """Defensive int cast. None / "" / "abc" / [] / bool → None.

    bool — guarded explicitly: int(True) == 1 silently matches candidate_id=1
    if LLM ever emits a JSON boolean. Floats are truncated (int(1.7) == 1) —
    treated as LLM intent error elsewhere via candidate_id check.
    """
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def _fallback_clarification(reason: str, *, question: str = "") -> ToolResult:
    """LLM выдал bad args — bounce на clarification вместо raise.

    reason — для логов/audit. question — опциональный override (по умолчанию
    общий «уточните»).
    """
    logger.warning("ai.tool_call.fallback reason=%s", reason)
    return ToolResult(
        action_type=ActionType.ASK_CLARIFICATION,
        action_data={
            "question": question or "Уточните, пожалуйста, что вы хотели бы найти?",
            "options": [],
        },
    )


# ─── handle_show_masters ──────────────────────────────────────────────────


def handle_show_masters(args: dict[str, Any], context: MasterContext) -> ToolResult:
    """Validate master_ids в context.candidate_ids, drop invalid silently.

    Anti-hallucination: LLM выдумал ID — фильтруем. Если все невалидные →
    fallback. Partial result (часть валидных) лучше чем dead chat turn.
    """
    raw_ids = args.get("master_ids") or []
    scores = args.get("match_scores") or []
    reasons = args.get("match_reasons") or []
    explanation = args.get("explanation") or ""

    valid_ids_normalized = [_safe_int(rid) for rid in raw_ids]
    valid_ids = [
        vid for vid in valid_ids_normalized
        if vid is not None and vid in context.candidate_ids
    ]

    if not valid_ids:
        return _fallback_clarification("show_masters_no_valid_ids")

    by_id = {c.id: c for c in context.candidates}
    masters: list[dict[str, Any]] = []
    for idx, mid in enumerate(valid_ids):
        c = by_id[mid]
        masters.append({
            "master": {
                "id": c.id,
                "name": c.name,
                "specialization": c.specialization,
                "services_preview": [name for _, name in c.services[:5]],
            },
            "match_score": scores[idx] if idx < len(scores) else None,
            "match_reasons": reasons[idx] if idx < len(reasons) else [],
        })

    return ToolResult(
        action_type=ActionType.SHOW_MASTERS,
        action_data={"masters": masters, "explanation": explanation},
    )


# ─── handle_show_slots ────────────────────────────────────────────────────


def handle_show_slots(args: dict[str, Any], context: MasterContext) -> ToolResult:
    """Validate (master_id, service_id) в context + ISO date.

    Реальный fetch слотов (YClients API / Ayla appointments) — в render
    layer консумера. Handler делает только валидацию args + проверку что
    мастер действительно оказывает выбранную услугу.
    """
    master_id = _safe_int(args.get("master_id"))
    service_id = _safe_int(args.get("service_id"))
    date_str = args.get("date") or ""

    if master_id is None or master_id not in context.candidate_ids:
        return _fallback_clarification("show_slots_invalid_master_id")
    if service_id is None or service_id not in context.candidate_service_ids:
        return _fallback_clarification("show_slots_invalid_service_id")
    try:
        target_date = date.fromisoformat(date_str)
    except (ValueError, TypeError):
        return _fallback_clarification("show_slots_invalid_date")

    # Cross-validation: master.services должен содержать service
    master = next((c for c in context.candidates if c.id == master_id), None)
    if master is None:
        return _fallback_clarification("show_slots_master_lost")
    master_service_ids = {sid for sid, _ in master.services}
    if service_id not in master_service_ids:
        return _fallback_clarification(
            "show_slots_master_does_not_offer_service",
            question="Этот мастер не оказывает данную услугу. Подобрать другого?",
        )

    return ToolResult(
        action_type=ActionType.SHOW_SLOTS,
        action_data={
            "master_id": master_id,
            "service_id": service_id,
            "date": target_date.isoformat(),
        },
    )


# ─── handle_confirm_booking ───────────────────────────────────────────────


def handle_confirm_booking(
    args: dict[str, Any],
    context: MasterContext,
    *,
    master_resolver: MasterResolver | None = None,
    service_resolver: ServiceResolver | None = None,
) -> ToolResult:
    """Validate (master_id, service_id) + ISO datetime.

    Не создаёт BookingRequest — это работа caller-side action service после
    клика клиента «✅ Да».

    master_resolver / service_resolver — optional callables для enrichment
    action_data полями master_name / service_name / price_from / duration_min
    (для рендера confirmation card). Если не предоставлены — поля None,
    caller дополняет в render layer.
    """
    master_id = _safe_int(args.get("master_id"))
    service_id = _safe_int(args.get("service_id"))
    datetime_str = args.get("datetime") or ""

    if master_id is None or master_id not in context.candidate_ids:
        return _fallback_clarification("confirm_booking_invalid_master_id")
    if service_id is None or service_id not in context.candidate_service_ids:
        return _fallback_clarification("confirm_booking_invalid_service_id")

    # Cross-validation: master.services должен содержать service (mirror handle_show_slots).
    # Без этого LLM trust boundary leak: (master_id=Анна, service_id=Бориса) проходит,
    # потому что service_id в global candidate_service_ids — но мастер его не оказывает.
    master = next((c for c in context.candidates if c.id == master_id), None)
    if master is None:
        return _fallback_clarification("confirm_booking_master_lost")
    master_service_ids = {sid for sid, _ in master.services}
    if service_id not in master_service_ids:
        return _fallback_clarification(
            "confirm_booking_master_does_not_offer_service",
            question="Этот мастер не оказывает данную услугу. Подобрать другого?",
        )

    try:
        slot = datetime.fromisoformat(datetime_str)
    except (ValueError, TypeError):
        return _fallback_clarification("confirm_booking_invalid_datetime")

    action_data: dict[str, Any] = {
        "master_id": master_id,
        "service_id": service_id,
        "datetime": slot.isoformat(),
        "master_name": None,
        "service_name": None,
        "price_from": None,
        "duration_min": None,
    }

    # Optional enrichment через resolvers. resolver-None semantics:
    # «мастер/услуга деактивированы между recommendation и confirmation» (race)
    # — отдельный fallback с конкретным user-facing question + отдельным reason
    # tag для ops-наблюдаемости.
    if master_resolver is not None:
        master_data = master_resolver(master_id)
        if master_data is None:
            return _fallback_clarification(
                "confirm_booking_master_unavailable",
                question="Этот мастер больше недоступен. Подобрать другого?",
            )
        action_data["master_name"] = master_data.get("name")

    if service_resolver is not None:
        service_data = service_resolver(service_id)
        if service_data is None:
            return _fallback_clarification(
                "confirm_booking_service_unavailable",
                question="Эта услуга временно недоступна. Подобрать другую?",
            )
        action_data["service_name"] = service_data.get("name")
        # price_from coercion: Decimal не JSON-serializable, а action_data уйдёт
        # в Message.action_data (JSONField). Stringify defensively — соответствует
        # behavior бота `str(service.price_from) if service.price_from else None`.
        price = service_data.get("price_from")
        action_data["price_from"] = str(price) if price is not None else None
        action_data["duration_min"] = service_data.get("duration_min")

    return ToolResult(
        action_type=ActionType.CONFIRM_BOOKING,
        action_data=action_data,
    )


# ─── handle_show_my_bookings ──────────────────────────────────────────────


def handle_show_my_bookings(args: dict[str, Any]) -> ToolResult:
    """Validate filter enum, default upcoming. Fetch — в render layer."""
    raw_filter = args.get("filter") or "upcoming"
    if raw_filter not in {"upcoming", "past", "all"}:
        raw_filter = "upcoming"
    return ToolResult(
        action_type=ActionType.SHOW_MY_BOOKINGS,
        action_data={"filter": raw_filter},
    )


# ─── handle_ask_clarification ─────────────────────────────────────────────


def handle_ask_clarification(args: dict[str, Any]) -> ToolResult:
    """Pass-through. Если LLM эмиттит пустой question → fallback с дефолтом."""
    question = (args.get("question") or "").strip()
    options = args.get("options") or []
    if not question:
        return _fallback_clarification("ask_clarification_empty_question")
    return ToolResult(
        action_type=ActionType.ASK_CLARIFICATION,
        action_data={"question": question, "options": options},
    )


# ─── dispatch_tool_call ───────────────────────────────────────────────────


def dispatch_tool_call(
    tool_call: Any,
    context: MasterContext,
    *,
    master_resolver: MasterResolver | None = None,
    service_resolver: ServiceResolver | None = None,
) -> ToolResult:
    """Главный диспетчер. Принимает OpenAI tool_call object, возвращает ToolResult.

    Безопасность: невалидный JSON в arguments / unknown tool name →
    fallback на ask_clarification, не raise.

    tool_call — OpenAI's tool_call с .function.name и .function.arguments
    (JSON-string). Duck-typed (Any) чтобы не зависеть от SDK-версии.
    """
    name = getattr(tool_call.function, "name", "") or ""
    raw_args = getattr(tool_call.function, "arguments", "") or "{}"

    try:
        args = json.loads(raw_args)
    except (json.JSONDecodeError, ValueError):
        return _fallback_clarification(f"invalid_json_arguments_in_{name}")

    if name == ActionType.SHOW_MASTERS:
        return handle_show_masters(args, context)
    if name == ActionType.SHOW_SLOTS:
        return handle_show_slots(args, context)
    if name == ActionType.CONFIRM_BOOKING:
        return handle_confirm_booking(
            args, context,
            master_resolver=master_resolver,
            service_resolver=service_resolver,
        )
    if name == ActionType.SHOW_MY_BOOKINGS:
        return handle_show_my_bookings(args)
    if name == ActionType.ASK_CLARIFICATION:
        return handle_ask_clarification(args)

    return _fallback_clarification(f"unknown_tool_{name}")
