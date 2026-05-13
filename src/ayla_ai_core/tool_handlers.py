"""Tool-call handlers — валидация LLM args + формирование action_data.

DRF-237 — извлечение из бота. DRF-238 — generic over ID type через injected
`id_parser` callable.

Handlers SIDE-EFFECT-FREE:
- Не делают network calls
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

ID parsing (DRF-238):
- Default `_safe_int` для int IDs (бот, backward compat).
- `_safe_uuid` для UUID IDs (Ayla).
- Custom parser передаётся через `dispatch_tool_call(..., id_parser=...)`.

Cross-validation в handle_show_slots / handle_confirm_booking — master.services
должен содержать service_id (LLM trust boundary layer 2 — даже если оба ID
валидны globally, проверяется что мастер реально оказывает услугу).

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
from uuid import UUID

from ayla_ai_core.context import SpecialistContext
from ayla_ai_core.tools import ActionType

__all__ = [
    "MasterResolver",
    "ServiceResolver",
    "ToolResult",
    "_safe_int",
    "_safe_uuid",
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
# duration_min (int | None), и (v0.7.0) optional `tenant_id` для cross-tenant
# guard. None если объект не найден.
#
# **v0.7.0 breaking change**: resolvers must accept a kwarg-only `tenant_id`:
#     def my_resolver(value: int, *, tenant_id: str) -> dict | None: ...
# The dispatcher passes ``context.tenant_id`` so the resolver can scope its
# ORM query (e.g., ``Master.objects.filter(id=value, tenant_id=tenant_id)``).
# Cross-tenant scoping at the resolver layer prevents a leak where the LLM
# emits a valid id from another tenant's history cached in the prompt.
MasterResolver = Callable[..., dict[str, Any] | None]
ServiceResolver = Callable[..., dict[str, Any] | None]


@dataclass(frozen=True)
class ToolResult:
    """Результат tool_handler — нормализованная пара (action_type, action_data).

    AIConcierge сохранит как Message.action_type / Message.action_data в
    persistence слое консумера.
    """

    action_type: str
    action_data: dict[str, Any]


# ─── ID parsers ───────────────────────────────────────────────────────────


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


def _safe_uuid(value: Any) -> UUID | None:
    """Defensive UUID cast. None / "" / non-UUID-string / non-string → None.

    Используется как `id_parser` для Ayla (SpecialistProfile.id). LLM эмиттит
    UUID как строку в JSON; UUID()-конструктор поднимает ValueError на не-UUID.
    """
    if value is None or value == "":
        return None
    if isinstance(value, UUID):
        return value
    if not isinstance(value, str):
        return None
    try:
        return UUID(value)
    except (ValueError, AttributeError):
        return None


# ─── Helpers ──────────────────────────────────────────────────────────────


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


def _assert_tenant_id_set(context: SpecialistContext[Any]) -> None:
    """B3 (v0.7.0): refuse to dispatch on a context lacking tenant_id.

    Multi-tenant isolation is a security boundary — silently allowing a
    context with empty/missing tenant_id was the v0.6.x footgun that let
    cross-tenant leakage slip past static checks. Fail loud at every
    handler entry so the bug surfaces in tests, not in production.
    """
    if not getattr(context, "tenant_id", ""):
        raise ValueError(
            "context.tenant_id required for tenant-scoped dispatch (v0.7.0)"
        )


# Sentinel attribute name. Setting it to True on a resolver callable opts that
# resolver out of the v0.7.2 mandatory-tenant_id check. Use only when the
# resolver genuinely cannot supply tenant_id (e.g., wrapping legacy code that
# returns plain rows); audit and remove the opt-out for any resolver that
# touches user-tenant data.
RESOLVER_SKIPS_TENANT_CHECK_ATTR = "__resolver_skips_tenant_check__"


def _check_resolver_tenant(
    resolved: dict[str, Any],
    *,
    expected_tenant_id: str,
    resolver: Any,
    resolver_kind: str,
) -> str | None:
    """v0.7.2 Sec-1: enforce that a resolver returned a row scoped to the
    caller's tenant.

    Returns ``None`` if the check passes; otherwise a short reason tag the
    caller appends to its fallback identifier. Three outcomes:

    1. **Pass** (return ``None``) — ``resolved["tenant_id"] == expected_tenant_id``.
    2. **Missing** (return ``"resolver_no_tenant_id"``) — resolver returned a
       dict without a ``tenant_id`` field. Was permissive in v0.7.0 (assumed
       resolver was scoped at the ORM layer); v0.7.2 treats it as a hard fail
       so the bug surfaces in tests rather than as a quiet cross-tenant leak.
    3. **Mismatch** (return ``"tenant_mismatch"``) — resolver returned a row
       belonging to a different tenant. Same fallback as v0.7.0.

    **Opt-out**: setting the ``__resolver_skips_tenant_check__`` attribute on
    the resolver callable to ``True`` makes the check a no-op (returns
    ``None`` immediately) and emits a single ``WARNING`` per call so it stays
    visible in audit logs. Reserved for resolvers wrapping legacy code that
    can't include tenant_id in its return; new resolvers MUST include
    tenant_id and not use this escape hatch.
    """
    if getattr(resolver, RESOLVER_SKIPS_TENANT_CHECK_ATTR, False):
        logger.warning(
            "ai.tool_call.resolver_skips_tenant_check resolver=%s kind=%s "
            "expected_tenant_id=%s — opt-out should be removed",
            getattr(resolver, "__name__", repr(resolver)),
            resolver_kind,
            expected_tenant_id,
        )
        return None

    tenant_id = resolved.get("tenant_id")
    if tenant_id is None:
        return "resolver_no_tenant_id"
    if tenant_id != expected_tenant_id:
        return "tenant_mismatch"
    return None


# ─── handle_show_masters ──────────────────────────────────────────────────


def handle_show_masters(
    args: dict[str, Any],
    context: SpecialistContext[Any],
    *,
    id_parser: Callable[[Any], Any] = _safe_int,
) -> ToolResult:
    """Validate master_ids в context.candidate_ids, drop invalid silently.

    Anti-hallucination: LLM выдумал ID — фильтруем. Если все невалидные →
    fallback. Partial result (часть валидных) лучше чем dead chat turn.

    id_parser: int для бота (default), UUID для Ayla, custom callable для
    других консумеров.
    """
    _assert_tenant_id_set(context)

    raw_ids = args.get("master_ids") or []
    scores = args.get("match_scores") or []
    reasons = args.get("match_reasons") or []
    explanation = args.get("explanation") or ""

    # B2 (v0.7.0): align scores/reasons with the LLM-emitted INDEX of each
    # id, not with the post-filter index. Filtering hallucinated IDs
    # previously misaligned the surviving entries — Anna (id=1) inherited
    # the score the model assigned to the hallucinated id=999. Carry
    # (original_idx, normalized_id) through filtering + dedup so each
    # surviving master maps back to its original score/reason slot.
    valid_ids_normalized = [id_parser(rid) for rid in raw_ids]
    valid_pairs: list[tuple[int, Any]] = []
    seen: set[Any] = set()
    for orig_idx, vid in enumerate(valid_ids_normalized):
        if vid is None or vid not in context.candidate_ids:
            continue
        # Dedup: LLM occasionally emits master_ids=[1, 1, 1]; render
        # exactly one card per master.
        if vid in seen:
            continue
        seen.add(vid)
        valid_pairs.append((orig_idx, vid))

    if not valid_pairs:
        return _fallback_clarification("show_masters_no_valid_ids")

    # v0.7.2 (DRF-677): O(1) lookups via pre-computed by_id (was a fresh
    # dict comprehension on every call). Built once at context construction.
    masters: list[dict[str, Any]] = []
    for orig_idx, mid in valid_pairs:
        c = context.by_id[mid]
        masters.append({
            "master": {
                "id": c.id,
                "name": c.name,
                "specialization": c.specialization,
                "services_preview": [name for _, name in c.services[:5]],
            },
            "match_score": scores[orig_idx] if orig_idx < len(scores) else None,
            "match_reasons": reasons[orig_idx] if orig_idx < len(reasons) else [],
        })

    return ToolResult(
        action_type=ActionType.SHOW_MASTERS,
        action_data={"masters": masters, "explanation": explanation},
    )


# ─── handle_show_slots ────────────────────────────────────────────────────


def handle_show_slots(
    args: dict[str, Any],
    context: SpecialistContext[Any],
    *,
    id_parser: Callable[[Any], Any] = _safe_int,
) -> ToolResult:
    """Validate (master_id, service_id) в context + ISO date.

    Реальный fetch слотов (YClients API / Ayla appointments) — в render
    layer консумера. Handler делает только валидацию args + проверку что
    мастер действительно оказывает выбранную услугу.
    """
    _assert_tenant_id_set(context)

    master_id = id_parser(args.get("master_id"))
    service_id = id_parser(args.get("service_id"))
    date_str = args.get("date") or ""

    if master_id is None or master_id not in context.candidate_ids:
        return _fallback_clarification("show_slots_invalid_master_id")
    if service_id is None or service_id not in context.candidate_service_ids:
        return _fallback_clarification("show_slots_invalid_service_id")
    try:
        target_date = date.fromisoformat(date_str)
    except (ValueError, TypeError):
        return _fallback_clarification("show_slots_invalid_date")

    # Cross-validation: master.services должен содержать service.
    # v0.7.2 (DRF-677): O(1) by_id lookup + pre-computed service_id_set on
    # each candidate. Was `next(... for c in candidates if c.id == master_id)`
    # + set-comprehension per call at O(N²) for N=1000.
    master = context.by_id.get(master_id)
    if master is None:
        return _fallback_clarification("show_slots_master_lost")
    if service_id not in master.service_id_set:
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
    context: SpecialistContext[Any],
    *,
    master_resolver: MasterResolver | None = None,
    service_resolver: ServiceResolver | None = None,
    id_parser: Callable[[Any], Any] = _safe_int,
) -> ToolResult:
    """Validate (master_id, service_id) + ISO datetime.

    Не создаёт BookingRequest — это работа caller-side action service после
    клика клиента «✅ Да».

    master_resolver / service_resolver — optional callables для enrichment
    action_data полями master_name / service_name / price_from / duration_min
    (для рендера confirmation card). Если не предоставлены — поля None,
    caller дополняет в render layer.

    **v0.7.0**: resolvers вызываются с kwarg ``tenant_id=context.tenant_id``
    для cross-tenant scoping. Если resolver возвращает dict с ключом
    ``tenant_id``, его значение сверяется с ``context.tenant_id`` — mismatch
    бросает в ASK_CLARIFICATION.
    """
    _assert_tenant_id_set(context)

    master_id = id_parser(args.get("master_id"))
    service_id = id_parser(args.get("service_id"))
    datetime_str = args.get("datetime") or ""

    if master_id is None or master_id not in context.candidate_ids:
        return _fallback_clarification("confirm_booking_invalid_master_id")
    if service_id is None or service_id not in context.candidate_service_ids:
        return _fallback_clarification("confirm_booking_invalid_service_id")

    # Cross-validation: master.services должен содержать service (mirror handle_show_slots).
    # Без этого LLM trust boundary leak: (master_id=Анна, service_id=Бориса) проходит,
    # потому что service_id в global candidate_service_ids — но мастер его не оказывает.
    # v0.7.2 (DRF-677): O(1) by_id + service_id_set (was O(N²) per call).
    master = context.by_id.get(master_id)
    if master is None:
        return _fallback_clarification("confirm_booking_master_lost")
    if service_id not in master.service_id_set:
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
        master_data = master_resolver(master_id, tenant_id=context.tenant_id)
        if master_data is None:
            return _fallback_clarification(
                "confirm_booking_master_unavailable",
                question="Этот мастер больше недоступен. Подобрать другого?",
            )
        # v0.7.2 Sec-1 (B3 follow-up): strict cross-tenant guard. Resolver
        # MUST return a dict with tenant_id; missing → fallback. Defends
        # against a stale prompt referencing a different tenant's master
        # that got past the candidate_ids check. Opt-out per resolver via
        # __resolver_skips_tenant_check__ = True (warns).
        reason = _check_resolver_tenant(
            master_data,
            expected_tenant_id=context.tenant_id,
            resolver=master_resolver,
            resolver_kind="master",
        )
        if reason is not None:
            return _fallback_clarification(
                f"confirm_booking_master_{reason}",
                question="Подбор обновился — давайте подберём заново?",
            )
        action_data["master_name"] = master_data.get("name")

    if service_resolver is not None:
        service_data = service_resolver(service_id, tenant_id=context.tenant_id)
        if service_data is None:
            return _fallback_clarification(
                "confirm_booking_service_unavailable",
                question="Эта услуга временно недоступна. Подобрать другую?",
            )
        reason = _check_resolver_tenant(
            service_data,
            expected_tenant_id=context.tenant_id,
            resolver=service_resolver,
            resolver_kind="service",
        )
        if reason is not None:
            return _fallback_clarification(
                f"confirm_booking_service_{reason}",
                question="Подбор обновился — давайте подберём заново?",
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
    context: SpecialistContext[Any],
    *,
    master_resolver: MasterResolver | None = None,
    service_resolver: ServiceResolver | None = None,
    id_parser: Callable[[Any], Any] = _safe_int,
) -> ToolResult:
    """Главный диспетчер. Принимает OpenAI tool_call object, возвращает ToolResult.

    Безопасность: невалидный JSON в arguments / unknown tool name →
    fallback на ask_clarification, не raise.

    tool_call — OpenAI's tool_call с .function.name и .function.arguments
    (JSON-string). Duck-typed (Any) чтобы не зависеть от SDK-версии.

    id_parser — функция для парсинга master_id/service_id из LLM args.
    Default `_safe_int` (бот). Для UUID-консумеров (Ayla) пере давать
    `_safe_uuid`.

    **v0.7.0**: ``context.tenant_id`` обязателен — пустое значение бросает
    ``ValueError`` (security boundary, fail-loud at entry).
    """
    _assert_tenant_id_set(context)

    name = getattr(tool_call.function, "name", "") or ""
    raw_args = getattr(tool_call.function, "arguments", "") or "{}"

    try:
        args = json.loads(raw_args)
    except (json.JSONDecodeError, ValueError):
        return _fallback_clarification(f"invalid_json_arguments_in_{name}")

    if name == ActionType.SHOW_MASTERS:
        return handle_show_masters(args, context, id_parser=id_parser)
    if name == ActionType.SHOW_SLOTS:
        return handle_show_slots(args, context, id_parser=id_parser)
    if name == ActionType.CONFIRM_BOOKING:
        return handle_confirm_booking(
            args, context,
            master_resolver=master_resolver,
            service_resolver=service_resolver,
            id_parser=id_parser,
        )
    if name == ActionType.SHOW_MY_BOOKINGS:
        return handle_show_my_bookings(args)
    if name == ActionType.ASK_CLARIFICATION:
        return handle_ask_clarification(args)

    return _fallback_clarification(f"unknown_tool_{name}")
