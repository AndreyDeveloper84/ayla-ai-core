"""Master context — Top-N кандидатов мастеров с реальными ID для anti-hallucination.

Адаптация из `mysite/maxbot/ai_context.py`. Pure-Python dataclasses без Django:
концепция и render-логика общая, фактический build (ORM-запрос) остаётся в
консумере (бот / Ayla backend) и инжектится в `AIConcierge` через
`context_builder` callable.

В DRF-238 будет добавлен generic ID-type (int → UUID) и multi-tenant scoping;
сейчас только базовая dataclass с int ID (как в боте Формулы).
"""
from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "MasterCandidate",
    "MasterContext",
    "render_summary_text",
]


@dataclass(frozen=True)
class MasterCandidate:
    """Один мастер для контекста LLM.

    services — список (service_id, name) для cross-validation в handle_show_slots
    (handler проверяет что выбранная услуга действительно принадлежит мастеру).
    """

    id: int
    name: str
    specialization: str
    services: list[tuple[int, str]]


@dataclass(frozen=True)
class MasterContext:
    """Bundle кандидатов + frozenset ID для O(1) anti-hallucination check.

    candidate_ids и candidate_service_ids — frozenset для быстрой проверки
    `master_id in context.candidate_ids` в каждом tool_handler. Без них LLM
    галлюцинации (придуманный master_id) пропускались бы дальше.

    summary_text — markdown-рендер для system_prompt с явными
    `master_id=N` / `service_id=N` (LLM использует эти ID в tool_call args).
    """

    candidates: list[MasterCandidate]
    candidate_ids: frozenset[int]
    candidate_service_ids: frozenset[int]
    summary_text: str


def render_summary_text(candidates: list[MasterCandidate]) -> str:
    """Компактный markdown-список с явными ID для system_prompt.

    Формат:
        - master_id=42 Анна Иванова (массаж):
            * service_id=10 массаж спины
            * service_id=11 лимфодренаж
            * +ещё 11

    Trade-off: ~50 токенов/мастера vs anti-hallucination win. Real-world
    incident 2026-04-27: gpt-4o-mini выдал service_id=1 (не существующий),
    handler сфолбекнулся на ask_clarification — клиент не увидел поломку.
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


def build_master_context_from_candidates(candidates: list[MasterCandidate]) -> MasterContext:
    """Helper для consumer-side ORM builders: возьми candidates, получи MasterContext.

    Bot/Ayla строят `list[MasterCandidate]` через свой ORM, потом передают сюда —
    получают frozenset'ы и summary бесплатно.
    """
    all_service_ids: set[int] = set()
    for c in candidates:
        all_service_ids.update(sid for sid, _ in c.services)
    return MasterContext(
        candidates=candidates,
        candidate_ids=frozenset(c.id for c in candidates),
        candidate_service_ids=frozenset(all_service_ids),
        summary_text=render_summary_text(candidates),
    )
