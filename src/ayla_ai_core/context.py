"""SpecialistContext — Top-N кандидатов с реальными ID для anti-hallucination.

Извлечено в DRF-237 как `MasterContext` (int IDs only). DRF-238 расширил до
generic `SpecialistContext[ID_T]` — int для бота Формулы (Master), UUID для
Ayla (SpecialistProfile). Multi-tenant scoping через optional `tenant_id`.

Pure-Python dataclasses без Django: концепция и render-логика общая, фактический
build (ORM-запрос) остаётся в консумере (бот / Ayla backend) и инжектится в
`AIConcierge` через `context_builder` callable.

Backward compat:
- `MasterContext` / `MasterCandidate` остаются как aliases на
  `SpecialistContext[int]` / `SpecialistCandidate[int]` — бот не сломается до
  DRF-243 миграции.
- `build_master_context_from_candidates` — alias на specialist-версию.

Wire format (JSON Schema field names в tools.py):
- `master_id` / `service_id` остаются неизменными — это было в боте 30+ дней,
  LLM trained behavior зависит. Только семантика типа меняется (int → str для UUID).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, TypeVar, runtime_checkable
from uuid import UUID

__all__ = [
    # Generic (preferred)
    "ID_T",
    # v0.8.0 (Arch-1 / DRF-685): structural-typing Protocol over context
    # shapes — lets non-booking domains (FAQ chunks, support tickets)
    # implement the contract without subclassing or aliasing SpecialistContext.
    "CandidateContext",
    "ItemT",
    # Backward compat aliases (bot-side, deprecated)
    "MasterCandidate",
    "MasterContext",
    "SpecialistCandidate",
    "SpecialistContext",
    "build_master_context_from_candidates",
    "build_specialist_context_from_candidates",
    "render_summary_text",
]


# ID-type generic. Bot Формулы использует int (Django Master.id), Ayla — UUID
# (SpecialistProfile.id). Также допустимо str (для UUID-as-string wire format).
ID_T = TypeVar("ID_T", int, UUID, str)

# Item-type generic (v0.8.0 / Arch-1 / DRF-685). The booking domain uses
# `SpecialistCandidate[ID_T]` here; FAQ-style consumers can plug in their
# own shape (e.g. a `KbChunk` dataclass) as long as it carries an `.id`
# attribute the dispatcher can read.
ItemT = TypeVar("ItemT")


@runtime_checkable
class CandidateContext(Protocol[ID_T, ItemT]):
    """**Structural typing contract** for an LLM tool-call context.

    v0.8.0 introduces this Protocol so the orchestrator can accept any
    object that walks like a context, not just the booking-domain
    :class:`SpecialistContext`. Non-booking consumers (FAQ skill,
    support-ticket skill) implement their own frozen dataclass with the
    same field names and pass it to :meth:`AIConcierge.send_message` via
    ``context_builder``.

    Required attributes (read-only):

    - ``candidates: list[ItemT]`` — the top-N items that the LLM is
      allowed to reference. ``ItemT.id`` must be of type ``ID_T``.
    - ``candidate_ids: frozenset[ID_T]`` — pre-computed set of valid ids
      for O(1) anti-hallucination check inside tool handlers.
    - ``summary_text: str`` — rendered markdown summary that the prompt
      builder splices into the system prompt slot (composer override
      `masters_summary` for the booking template; a new section name
      for other domains).
    - ``tenant_id: str`` — multi-tenant isolation boundary, required.

    Optional attributes (booking-domain only — Protocol intentionally
    omits them so non-booking consumers don't have to provide):

    - ``candidate_service_ids: frozenset[ID_T]``
    - ``by_id: dict[ID_T, ItemT]``

    The Protocol is :func:`runtime_checkable`, so the orchestrator can do
    ``isinstance(ctx, CandidateContext)`` to validate consumer-supplied
    objects without requiring inheritance from a concrete class.
    """

    candidates: list[ItemT]
    candidate_ids: frozenset[ID_T]
    summary_text: str
    tenant_id: str


@dataclass(frozen=True)
class SpecialistCandidate[ID_T: (int, UUID, str)]:
    """Один specialist для контекста LLM.

    services — список (service_id, name) для cross-validation в handle_show_slots
    и handle_confirm_booking (handler проверяет что выбранная услуга действительно
    принадлежит specialist'у — anti-hallucination layer 2).

    v0.7.2 (DRF-677): ``service_id_set`` — frozenset of ``sid`` values from
    ``services`` for O(1) membership checks in handlers. Populated by
    :func:`build_specialist_context_from_candidates`; for direct construction
    pass either an explicit frozenset or let the default `field(default_factory=...)`
    compute it (deferred to consumer; here we keep an empty default so we
    don't break v0.7.0 callers that construct via direct keyword args).
    """

    id: ID_T
    name: str
    specialization: str
    services: list[tuple[ID_T, str]]
    # NEW in v0.7.2: pre-computed for O(1) `sid in candidate.service_id_set` in
    # handlers. Defaults to empty frozenset; __post_init__ auto-populates
    # from `services` so direct-constructor callers stay backward-compatible
    # without manually computing it.
    service_id_set: frozenset[ID_T] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        # v0.7.2 (DRF-677): auto-populate when caller didn't pass an explicit
        # service_id_set. Frozen dataclass: must use object.__setattr__.
        if not self.service_id_set and self.services:
            object.__setattr__(
                self, "service_id_set", frozenset(sid for sid, _ in self.services)
            )


@dataclass(frozen=True)
class SpecialistContext[ID_T: (int, UUID, str)]:
    """Bundle кандидатов + frozenset ID для O(1) anti-hallucination check.

    candidate_ids и candidate_service_ids — frozenset для быстрой проверки
    `master_id in context.candidate_ids` в каждом tool_handler.

    summary_text — markdown-рендер для system_prompt с явными
    `master_id=N` / `service_id=N` (LLM использует эти ID в tool_call args).
    Wire format остаётся `master_id` для backward compat.

    tenant_id — multi-tenant scope. **Required since v0.7.0** (was optional
    in v0.6.x). Multi-tenant isolation is a security boundary, not a feature
    flag — defaulting to None silently allowed cross-tenant leakage in
    consumers that forgot to pass it. Single-tenant consumers pass a stable
    sentinel (e.g., `"formula-tela"`) to satisfy the contract.
    """

    candidates: list[SpecialistCandidate[ID_T]]
    candidate_ids: frozenset[ID_T]
    candidate_service_ids: frozenset[ID_T]
    summary_text: str
    tenant_id: str
    # NEW in v0.7.2: precomputed `{candidate.id -> candidate}` for O(1)
    # handler lookups. Defaults to empty dict; __post_init__ auto-populates
    # from `candidates` so direct-constructor callers stay backward-compatible.
    by_id: dict[ID_T, SpecialistCandidate[ID_T]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # v0.7.2 (DRF-677): auto-populate when caller didn't pass an explicit
        # by_id. Frozen dataclass: must use object.__setattr__.
        if not self.by_id and self.candidates:
            object.__setattr__(
                self, "by_id", {c.id: c for c in self.candidates}
            )


def render_summary_text[ID_T: (int, UUID, str)](candidates: list[SpecialistCandidate[ID_T]]) -> str:
    """Компактный markdown-список с явными ID для system_prompt.

    Формат:
        - master_id=42 Анна Иванова (массаж):
            * service_id=10 массаж спины
            * service_id=11 лимфодренаж
            * +ещё 11

    Trade-off: ~50 токенов/specialist vs anti-hallucination win. Real-world
    incident 2026-04-27 (бот Формулы): gpt-4o-mini выдал service_id=1 (не
    существующий), handler сфолбекнулся на ask_clarification — клиент молча
    получил уточняющий вопрос, не error.

    Wire format: для UUID-консумеров (Ayla) c.id будет `UUID('abc-123-...')`,
    str(c.id) даёт "abc-123-...". Для int (бот) — "42". В обоих случаях
    LLM получает строку и эмиттит её в tool_call.
    """
    if not candidates:
        return "(нет активных мастеров — записаться можно только через менеджера)"

    lines: list[str] = []
    for c in candidates:
        spec = f" ({c.specialization})" if c.specialization else ""
        lines.append(f"- master_id={c.id} {c.name}{spec}:")
        top_services = c.services[:5]
        for sid, name in top_services:
            lines.append(f"    * service_id={sid} {name}")
        if len(c.services) > 5:
            lines.append(f"    * +ещё {len(c.services) - 5} услуг")
    return "\n".join(lines)


def build_specialist_context_from_candidates[ID_T: (int, UUID, str)](
    candidates: list[SpecialistCandidate[ID_T]],
    *,
    tenant_id: str,
) -> SpecialistContext[ID_T]:
    """Helper для consumer-side ORM builders: возьми candidates, получи SpecialistContext.

    Бот / Ayla строят `list[SpecialistCandidate[ID_T]]` через свой ORM, потом
    передают сюда — получают frozenset'ы и summary бесплатно.

    tenant_id — **required since v0.7.0** (was optional kwarg in v0.6.x).
    Empty string is rejected — multi-tenant scoping is a security boundary
    and a typo'd empty literal is exactly the regression v0.7.0 closes.
    """
    if not tenant_id:
        raise ValueError(
            "tenant_id is required (v0.7.0); pass a non-empty stable identifier"
        )
    # v0.7.2 (DRF-677): pre-compute O(1) lookup structures. Replaces a
    # linear scan in each handler at N=20 (1-2ms) -> O(N²) at N=1000
    # (50-100ms) — breached per-turn budget at SaaS catalog scale.
    # Re-wrap candidates with their service_id_set so handler can do
    # `service_id in candidate.service_id_set` instead of rebuilding the
    # set on every tool call.
    candidates_with_sets: list[SpecialistCandidate[ID_T]] = [
        SpecialistCandidate(
            id=c.id,
            name=c.name,
            specialization=c.specialization,
            services=c.services,
            service_id_set=frozenset(sid for sid, _ in c.services),
        )
        for c in candidates
    ]
    all_service_ids: set[ID_T] = set()
    for c in candidates_with_sets:
        all_service_ids.update(c.service_id_set)
    return SpecialistContext(
        candidates=candidates_with_sets,
        candidate_ids=frozenset(c.id for c in candidates_with_sets),
        candidate_service_ids=frozenset(all_service_ids),
        summary_text=render_summary_text(candidates_with_sets),
        tenant_id=tenant_id,
        by_id={c.id: c for c in candidates_with_sets},
    )


# ─── Backward compat aliases (DRF-237 → DRF-238) ──────────────────────────
# Bot Формулы использует MasterCandidate / MasterContext до DRF-243 миграции.
# После DRF-243 эти aliases можно удалить.

MasterCandidate = SpecialistCandidate[int]
"""DEPRECATED: используй SpecialistCandidate[int]. Alias для backward compat с ботом."""

MasterContext = SpecialistContext[int]
"""DEPRECATED: используй SpecialistContext[int]. Alias для backward compat с ботом."""


def build_master_context_from_candidates(
    candidates: list[SpecialistCandidate[int]],
    *,
    tenant_id: str,
) -> SpecialistContext[int]:
    """DEPRECATED: используй build_specialist_context_from_candidates.

    **v0.7.0**: tenant_id теперь обязательный — single-tenant консумеры
    передают стабильный sentinel (например, "formula-tela").
    """
    return build_specialist_context_from_candidates(candidates, tenant_id=tenant_id)
