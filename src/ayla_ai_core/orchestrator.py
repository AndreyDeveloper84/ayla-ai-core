"""AIConcierge — главный chat-pipeline orchestrator.

Извлечено в DRF-237 из `mysite/maxbot/ai_concierge.py` (production-tested
30+ days). См. `docs/BOT_CODE_AUDIT_2026-04.md` Section 1.3 (architectural
pattern) и Section 4 (Variant C+ shared package strategy).

Pipeline на каждый user-message:
1. store.resolve_active_conversation(user_key) → existing OR new
2. store.save_message(role=user) → user_msg.id
3. context_builder() → MasterContext (Top-N с реальными ID)
4. store.load_recent_history(conv, exclude_id=user_msg.id, limit) → list[Message]
5. prompt_renderer(master_context) → system_prompt str
6. _compose_messages(system, history, user_text) → list[dict]
7. openai_client.chat.completions.create(tools=TOOL_DEFINITIONS) async
8. measure latency_ms
9. parse completion: content + (optional tool_call → dispatch_tool_call → action)
10. store.save_message(role=assistant, action_type, action_data, tool_call,
    tokens_in/out, latency_ms)
11. return ChatResponseDTO

Не делает рендер UI (это в консумере: ai_ui для бота, REST serializer для Ayla)
и не пишет BookingRequest (это в action_service консумера) — только Conversation
+ Message через injected store.

DI / abstraction:
- `openai_client`: AsyncOpenAI — caller injects
- `store: ConversationStore` — Protocol с 3 методами (resolve_active_conversation,
  save_message, load_recent_history) — каждый консумер реализует против своего
  ORM (Django models в боте, отдельные модели в Ayla DRF-240)
- `context_builder: Callable[[], MasterContext]` — caller-side ORM query
- `prompt_renderer: Callable[[MasterContext], str]` — caller-side prompt
  template (DRF-239 параметризует с brand_voice)
- `master_resolver / service_resolver` — opt callable для confirm_booking
  enrichment

Sync vs Async:
- `store` методы — sync (вызываются через asgiref.sync_to_async внутри)
- `openai_client.chat.completions.create` — async native
- `context_builder` — sync OR async (определяется helper'ом)
"""
from __future__ import annotations

import functools
import inspect
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from asgiref.sync import sync_to_async

from ayla_ai_core.context import SpecialistContext
from ayla_ai_core.observability import (
    current_tenant_id,
    scope_frozen_now,
    scope_tenant_id,
)
from ayla_ai_core.tool_handlers import (
    MasterResolver,
    ServiceResolver,
    ToolResult,
    _safe_int,
    dispatch_tool_call,
)
from ayla_ai_core.tools import TOOL_DEFINITIONS, ActionType

__all__ = [
    "DEFAULT_HISTORY_LIMIT",
    "DEFAULT_MODEL_NAME",
    "AIConcierge",
    "ChatResponseDTO",
    "ConversationStore",
    "MessageRole",
    "ToolDispatcher",
]


# Custom tool dispatcher hook (DRF-241 / 0.6.0).
# Consumers with their own wire-format (Ayla: show_specialists vs shared
# show_masters) inject a callable that takes the OpenAI tool_call object
# + the specialist_context and returns a ToolResult. When None, AIConcierge
# falls back to the bundled `dispatch_tool_call` — backward-compatible for
# every existing consumer (default behaviour unchanged).
ToolDispatcher = Callable[[Any, SpecialistContext[Any]], ToolResult]


logger = logging.getLogger("ayla_ai_core.orchestrator")

DEFAULT_HISTORY_LIMIT = 10
DEFAULT_MODEL_NAME = "gpt-4o-mini"

# v0.7.3 / Obs-4 (DRF-684). Default per-turn history-token budget. The
# count-based DEFAULT_HISTORY_LIMIT is still applied at the load_recent_history
# layer (DB-side); the token budget is the second cap, enforced inside
# _compose_messages so a verbose 10-message history can't blow the LLM
# context (cost ceiling, not correctness).
DEFAULT_HISTORY_TOKEN_BUDGET = 4000

# v0.7.4: per-message OpenAI envelope overhead constants. Used by the
# token-budget guard so the budget reflects actual completion cost, not
# just raw content. See _compose_messages for rationale.
_MSG_ENVELOPE_TOKENS = 4
_PRIMER_OVERHEAD_TOKENS = 3


# v0.7.4 (DRF-684 follow-up): cache is keyed on model_name, not process-global.
# Pre-fix: a multi-model deployment (one AIConcierge on gpt-4o-mini, another on
# claude-haiku) silently reused whichever encoder loaded first → wrong token
# counts on the second model. lru_cache(maxsize=8) keys per-model so each
# model gets its own encoder, sized to the realistic number of models a
# single process serves.
_tiktoken_unavailable_warned = False


def _warn_tiktoken_missing_once() -> None:
    """Emit the tiktoken-missing fallback warning exactly once per process."""
    global _tiktoken_unavailable_warned
    if _tiktoken_unavailable_warned:
        return
    logger.warning(
        "tiktoken not installed — history token budget falling back "
        "to a 4-chars-per-token heuristic. Install with "
        "`uv pip install ayla-ai-core[tiktoken]` for accurate counting.",
        extra={"tenant_id": ""},
    )
    _tiktoken_unavailable_warned = True


@functools.lru_cache(maxsize=8)
def _get_encoder(model_name: str) -> Any:
    """Return a cached tiktoken encoder for ``model_name``, or ``None``
    if tiktoken is not installed. v0.7.4 (Code-Reviewer P0 fix): keyed
    per model so multi-model processes don't share a single encoder.
    """
    try:
        import tiktoken  # type: ignore[import-not-found]
    except ImportError:
        _warn_tiktoken_missing_once()
        return None
    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        # Model not recognised by tiktoken (custom model name etc.) —
        # fall back to the universal cl100k_base which covers gpt-4 family.
        return tiktoken.get_encoding("cl100k_base")


def _estimate_tokens(text: str, encoder: Any) -> int:
    """Count tokens for ``text``. Uses tiktoken when ``encoder`` is set,
    otherwise the 4-chars-per-token heuristic (good to ~10% for Latin /
    Cyrillic mix in our domain — exact value isn't critical because the
    budget guard is a cost ceiling, not a hard context-window cap).
    """
    if encoder is None:
        # Heuristic: ~4 chars/token for mixed Latin + Cyrillic. Rough but
        # bounded — if we get within 20% of the budget we're successfully
        # protecting per-turn cost from chatty users.
        return max(1, len(text) // 4)
    return len(encoder.encode(text))


class MessageRole:
    """Role wire-format. String values чтобы Django TextChoices совпадали."""

    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    SYSTEM = "system"


@dataclass(frozen=True)
class ChatResponseDTO:
    """Что AIConcierge возвращает caller'у (handler-у консумера).

    conversation_id — для caller'а чтобы продолжать тот же диалог.
    content — текст assistant-а (может быть пустым если LLM сразу tool_call'нул).
    action_type / action_data — primary tool action для UI рендера. None если LLM
    ответил pure-text (small-talk).

    extra_actions (v0.7.2 / DRF-678) — additional tool actions when the LLM
    emits multiple parallel tool_calls (gpt-4o + ``parallel_tool_calls=True``).
    Each entry is ``{"action_type": str, "action_data": dict}``. ``None`` when
    the response had 0 or 1 tool_call (backward-compat with v0.7.0/v0.6.x).
    Primary (action_type/action_data) is the first non-clarification result;
    extras carry the rest in their original order. Callers that don't render
    compound actions can safely ignore this field.
    """

    conversation_id: Any  # UUID для бота / Ayla, может быть int в legacy
    content: str
    action_type: str | None
    action_data: dict[str, Any] | None
    extra_actions: list[dict[str, Any]] | None = None
    # NEW in v0.7.3 (DRF-682): telemetry surfaced as data, not just in logs.
    # All five default to a neutral zero/empty so v0.7.2 constructor sites
    # (positional or partial-kwarg) keep compiling — only consumers that
    # opt into reading them see the measured values.
    latency_ms: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    model: str = ""
    provider: str = ""


@runtime_checkable
class ConversationStore(Protocol):
    """Persistence backend для AIConcierge.

    Структурный typing — консумер передаёт любой объект с этими 3 методами.
    Bot реализует через `services_app.models` (Django ORM); Ayla — через свои
    Conversation/Message модели (DRF-240).

    Возвращаемые типы — duck-typed. Conversation.id любого типа (UUID/int);
    Message — любой объект с .id, .role, .content (для рендера в LLM history).
    """

    def resolve_active_conversation(self, user_key: Any) -> Any:
        """Существующий active OR создать новый. Один active на user_key."""
        ...

    def save_message(
        self,
        conversation: Any,
        *,
        role: str,
        content: str,
        action_type: str = "",
        action_data: dict | None = None,
        tool_call: dict | None = None,
        tool_call_id: str = "",
        tokens_in: int = 0,
        tokens_out: int = 0,
        latency_ms: int | None = None,
    ) -> Any:
        """Сохранить Message + обновить conversation.last_message_at."""
        ...

    def load_recent_history(
        self,
        conversation: Any,
        *,
        exclude_id: Any | None = None,
        limit: int = DEFAULT_HISTORY_LIMIT,
    ) -> list[Any]:
        """Last N messages в хронологическом порядке (ASC) для LLM context.

        Pattern: берём last N (DESC), reverse — корректно сохраняет хронологию
        даже когда сообщений > N (старые вытесняются).
        """
        ...


# ─── Helpers ──────────────────────────────────────────────────────────────


def _compose_messages(
    *,
    system_prompt: str,
    history: list[Any],
    user_text: str,
    token_budget: int | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
) -> list[dict[str, Any]]:
    """OpenAI-format messages: system + history + new user.

    history elements должны иметь .role и .content атрибуты (duck-typed).
    Tool/system messages в истории пропускаются — на текущей итерации
    multi-step tool-use не делается.

    v0.7.3 (DRF-684): when ``token_budget`` is set, history is walked
    newest-first and accumulates until the budget is exhausted; older
    messages drop. ``system_prompt`` and ``user_text`` are always
    included (their cost is independent of replay history). When
    ``token_budget`` is ``None`` the count-based pre-filter remains the
    only cap — backward-compatible with v0.7.2.
    """
    encoder = _get_encoder(model_name) if token_budget is not None else None
    # Pre-walk newest-first to pick the longest tail that fits within the
    # budget. Replay an oldest-first compose afterwards so OpenAI sees the
    # correct chronological order.
    #
    # v0.7.4 (DRF-684 follow-up): account for OpenAI's per-message envelope
    # overhead. Constants live at module level (see _MSG_ENVELOPE_TOKENS,
    # _PRIMER_OVERHEAD_TOKENS) — every chat-completions request pays ~4
    # tokens per message + ~3 for the primer; without this the budget
    # undershoots by ~10% on a 10-message history.
    filtered_history: list[Any]
    if token_budget is not None:
        # Account for the system prompt + primer before the budget is spent
        # on history. Without this, system_prompt slips past the cap.
        used = (
            _PRIMER_OVERHEAD_TOKENS
            + _estimate_tokens(system_prompt, encoder)
            + _MSG_ENVELOPE_TOKENS
            + _estimate_tokens(user_text, encoder)
            + _MSG_ENVELOPE_TOKENS
        )
        kept: list[Any] = []
        for h in reversed(history):
            h_role = getattr(h, "role", None)
            if h_role not in (MessageRole.USER, MessageRole.ASSISTANT):
                continue
            content = getattr(h, "content", "") or ""
            tokens = _estimate_tokens(content, encoder) + _MSG_ENVELOPE_TOKENS
            if used + tokens > token_budget:
                break
            kept.append(h)
            used += tokens
        kept.reverse()
        filtered_history = kept
    else:
        filtered_history = history

    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    for h in filtered_history:
        h_role = getattr(h, "role", None)
        if h_role not in (MessageRole.USER, MessageRole.ASSISTANT):
            continue
        content = getattr(h, "content", "") or ""
        # B1 (v0.7.0): OpenAI rejects assistant turns with empty content and
        # no tool_calls. Such records appear when a prior turn invoked a tool
        # without emitting text — keeping them poisons every subsequent
        # completion in the conversation. Preserve assistant turns that DO
        # carry tool_calls (those carry the function-call payload OpenAI
        # needs to maintain the call/response pairing).
        if h_role == MessageRole.ASSISTANT and not content:
            tool_calls = getattr(h, "tool_calls", None)
            if not tool_calls:
                continue
        messages.append({"role": h_role, "content": content})
    messages.append({"role": MessageRole.USER, "content": user_text})
    return messages


def _parse_completion(
    completion: Any,
    specialist_context: SpecialistContext[Any],
    *,
    master_resolver: MasterResolver | None,
    service_resolver: ServiceResolver | None,
    id_parser: Callable[[Any], Any],
    tool_dispatcher: ToolDispatcher | None = None,
) -> tuple[str, dict | None, str | None, dict | None, list[dict[str, Any]] | None]:
    """Returns (content, primary_tool_call_raw, action_type, action_data, extra_actions).

    `tool_dispatcher` (DRF-241 / 0.6.0) — opt callable that takes
    (tool_call, specialist_context) and returns ToolResult. Consumers with
    their own wire-format (Ayla `show_specialists` vs shared `show_masters`)
    inject one to keep their own dispatch + handlers without forking the
    orchestrator. When None, the bundled `dispatch_tool_call` runs — every
    existing consumer keeps its current behaviour.

    **v0.7.2 (DRF-678)**: dispatches **every** entry in ``tool_calls``, not
    just ``[0]``. gpt-4o defaults ``parallel_tool_calls=True`` and routinely
    emits 2-3 calls — v0.7.0 silently dropped tail items. Merge strategy:

    1. First result whose ``action_type != ask_clarification`` becomes the
       primary (returned as ``action_type`` / ``action_data``).
    2. All other dispatched results are returned in ``extra_actions`` in the
       order the LLM emitted them. Consumers that only render the primary
       action can ignore this field — single-call responses still yield
       ``extra_actions=None`` exactly as v0.7.0.
    3. ``primary_tool_call_raw`` is the raw payload of the primary call (the
       one whose ToolResult became primary) so persistence keeps a stable
       pointer to "which call did we render".
    """
    msg = completion.choices[0].message
    tool_calls = getattr(msg, "tool_calls", None)
    content = msg.content or ""

    if not tool_calls:
        return content, None, None, None, None

    # Dispatch every tool_call in order. Handlers are side-effect-free
    # (validation + action_data formation only), so running all of them is
    # safe even if we end up presenting just one.
    dispatched: list[tuple[Any, Any]] = []  # (tool_call, ToolResult)
    for tc in tool_calls:
        if tool_dispatcher is not None:
            result = tool_dispatcher(tc, specialist_context)
        else:
            result = dispatch_tool_call(
                tc, specialist_context,
                master_resolver=master_resolver,
                service_resolver=service_resolver,
                id_parser=id_parser,
            )
        dispatched.append((tc, result))

    # "First non-clarification wins" — clarifications are fallback noise from
    # hallucinated ids etc.; promote the first concrete action ahead of them
    # so a (clarification, show_masters) pair from a confused gpt-4o still
    # renders the masters card.
    primary_idx = next(
        (i for i, (_, r) in enumerate(dispatched)
         if r.action_type != ActionType.ASK_CLARIFICATION),
        0,
    )
    primary_tc, primary_result = dispatched[primary_idx]

    primary_raw = {
        "id": getattr(primary_tc, "id", ""),
        "name": getattr(primary_tc.function, "name", ""),
        "arguments": getattr(primary_tc.function, "arguments", ""),
    }

    extra_actions: list[dict[str, Any]] | None
    if len(dispatched) == 1:
        # Backward-compat: single-call responses match v0.7.0 shape exactly.
        extra_actions = None
    else:
        extra_actions = [
            {"action_type": r.action_type, "action_data": r.action_data}
            for i, (_, r) in enumerate(dispatched)
            if i != primary_idx
        ]
        logger.info(
            "ai_concierge.parallel_tool_calls count=%d primary=%s",
            len(dispatched), primary_result.action_type,
            extra={"tenant_id": specialist_context.tenant_id},
        )

    return content, primary_raw, primary_result.action_type, primary_result.action_data, extra_actions


async def _maybe_await(value: Any) -> Any:
    """Если value — awaitable, await; иначе вернуть как есть.

    Позволяет context_builder быть sync (как в боте) ИЛИ async (если консумер
    хочет пускать ORM в thread pool).
    """
    if inspect.isawaitable(value):
        return await value
    return value


# ─── AIConcierge ──────────────────────────────────────────────────────────


class AIConcierge:
    """Главный orchestrator — async, stateless (state injected через DI).

    Один instance переиспользуется для всех user_key (state живёт в `store`).
    Создаётся один раз в startup консумера, send_message вызывается на каждый
    incoming message.
    """

    def __init__(
        self,
        *,
        openai_client: Any,
        store: ConversationStore,
        context_builder: Callable[[], Any],
        model_name: str = DEFAULT_MODEL_NAME,
        history_limit: int = DEFAULT_HISTORY_LIMIT,
        history_token_budget: int | None = DEFAULT_HISTORY_TOKEN_BUDGET,
        tool_definitions: list[dict] | None = None,
        id_parser: Callable[[Any], Any] = _safe_int,
        tool_dispatcher: ToolDispatcher | None = None,
    ) -> None:
        """Construct AIConcierge.

        openai_client: AsyncOpenAI — async OpenAI client (или совместимый).
        store: реализация ConversationStore (sync методы; будут обёрнуты в
            sync_to_async автоматически).
        context_builder: callable() → SpecialistContext (sync OR async). Должен
            быть дёшевым — вызывается на каждый turn.
        model_name: OpenAI model id. Default gpt-4o-mini (production-tested
            в боте Формулы).
        history_limit: max messages из истории для LLM context. Default 10.
        tool_definitions: override TOOL_DEFINITIONS. Default — int IDs.
            Для UUID consumers — `build_tool_definitions("string")`.
        id_parser: функция парсинга master_id/service_id из LLM args.
            Default `_safe_int` (бот). Для UUID-консумеров (Ayla) передавать
            `_safe_uuid`. См. tool_handlers.py.
        tool_dispatcher: opt DI-hook (DRF-241 / 0.6.0). Callable
            (tool_call, context) → ToolResult. Консумеры с собственным
            wire-format (Ayla: `show_specialists`/`specialist_id` vs shared
            `show_masters`/`master_id`) инжектят свой dispatcher и держат
            локальные tools.py + tool_handlers.py — без форка orchestrator.
            None (default) → bundled `dispatch_tool_call`. В этом случае
            id_parser / master_resolver / service_resolver применяются
            как раньше.
        """
        self._openai_client = openai_client
        self._store = store
        self._context_builder = context_builder
        self._model_name = model_name
        self._history_limit = history_limit
        self._history_token_budget = history_token_budget
        self._tool_definitions = tool_definitions or TOOL_DEFINITIONS
        self._id_parser = id_parser
        self._tool_dispatcher = tool_dispatcher

        # async-обёртки для sync ORM-методов store
        self._resolve_conv = sync_to_async(store.resolve_active_conversation)
        self._save_message = sync_to_async(store.save_message)
        self._load_history = sync_to_async(store.load_recent_history)

    async def send_message(
        self,
        *,
        user_key: Any,
        message_text: str,
        prompt_renderer: Callable[[SpecialistContext[Any]], str],
        master_resolver: MasterResolver | None = None,
        service_resolver: ServiceResolver | None = None,
        frozen_now: datetime | None = None,
    ) -> ChatResponseDTO:
        """Один turn AI Concierge: user message → assistant response с opt action.

        user_key: key для store.resolve_active_conversation (обычно BotUser/User
            instance, ID — зависит от store impl).
        message_text: текст от пользователя.
        prompt_renderer: callable(specialist_context) → system_prompt string.
            Caller инжектит свои template + state (today, client_name,
            bookings_count). DRF-239 заменит на BrandVoiceConfig-based
            renderer.
        master_resolver / service_resolver: opt enrichment для
            confirm_booking action_data (master_name, service_name,
            price_from, duration_min).

        Caller (handler консумера) ловит exceptions от openai_client и
        graceful'но реагирует (LLM_GIVEUP_MESSAGE → BotInquiry для бота,
        500 + Sentry для Ayla REST).
        """
        # 1. Resolve conversation
        conversation = await self._resolve_conv(user_key)

        # 2. Save user message (для exclude из history)
        user_msg = await self._save_message(
            conversation,
            role=MessageRole.USER,
            content=message_text,
        )

        # 3. Build context
        # Sync builder (типичный case бота: Django ORM query) — оборачиваем в
        # sync_to_async чтобы ORM не блокировал event loop / не ронял
        # SynchronousOnlyOperation. Async builder вызываем напрямую +
        # _maybe_await на результат (для совместимости с обоими стилями).
        if inspect.iscoroutinefunction(self._context_builder):
            specialist_context_raw = await self._context_builder()
        else:
            specialist_context_raw = await sync_to_async(self._context_builder)()
        specialist_context = await _maybe_await(specialist_context_raw)
        if not isinstance(specialist_context, SpecialistContext):
            raise TypeError(
                f"context_builder must return SpecialistContext, got {type(specialist_context)}"
            )

        # v0.7.3 (DRF-681): bind tenant_id to the ContextVar so every log
        # record emitted by ayla — including from tool_handlers helpers that
        # don't take context as a param — carries the field automatically.
        # v0.7.3 (DRF-683): also bind frozen_now (or None) so prompt
        # renderers + history filters can read the replay clock.
        async with (
            scope_tenant_id(specialist_context.tenant_id),
            scope_frozen_now(frozen_now),
        ):
            # 4. Load recent history
            history = await self._load_history(
                conversation,
                exclude_id=getattr(user_msg, "id", None),
                limit=self._history_limit,
            )

            # 5. Render system prompt
            system_prompt = prompt_renderer(specialist_context)

            # 6. Compose for OpenAI
            # v0.7.3 (DRF-684): token-budget guard caps per-turn history cost.
            llm_messages = _compose_messages(
                system_prompt=system_prompt,
                history=history,
                user_text=message_text,
                token_budget=self._history_token_budget,
                model_name=self._model_name,
            )

            # 7. Call OpenAI с tools
            started = time.monotonic()
            completion = await self._openai_client.chat.completions.create(
                model=self._model_name,
                messages=llm_messages,
                tools=self._tool_definitions,
            )
            latency_ms = int((time.monotonic() - started) * 1000)

            # 8. Parse: content + opt action(s).
            # v0.7.2 (DRF-678): _parse_completion now returns an extra
            # `extra_actions` slot for parallel tool_calls beyond the primary.
            content, tool_call_raw, action_type, action_data, extra_actions = _parse_completion(
                completion, specialist_context,
                master_resolver=master_resolver,
                service_resolver=service_resolver,
                id_parser=self._id_parser,
                tool_dispatcher=self._tool_dispatcher,
            )

            # 9. Telemetry
            usage = getattr(completion, "usage", None)
            tokens_in = getattr(usage, "prompt_tokens", 0) if usage else 0
            tokens_out = getattr(usage, "completion_tokens", 0) if usage else 0

            # 10. Save assistant message
            await self._save_message(
                conversation,
                role=MessageRole.ASSISTANT,
                content=content,
                action_type=action_type or "",
                action_data=action_data,
                tool_call=tool_call_raw,
                tool_call_id=(tool_call_raw or {}).get("id", "") if tool_call_raw else "",
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                latency_ms=latency_ms,
            )

            logger.info(
                "ai_concierge: conv=%s action=%s tokens=%d/%d latency=%dms",
                getattr(conversation, "id", "?"),
                action_type or "text",
                tokens_in, tokens_out, latency_ms,
                extra={"tenant_id": current_tenant_id()},
            )

            return ChatResponseDTO(
                conversation_id=getattr(conversation, "id", None),
                content=content,
                action_type=action_type,
                action_data=action_data,
                extra_actions=extra_actions,
                # v0.7.3 (DRF-682): surface telemetry as DTO fields so
                # consumers can build dashboards from return values
                # (instead of grepping log strings).
                latency_ms=latency_ms,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                model=self._model_name,
                provider="openai",  # v0.8.0 will multiplex via L-track router
            )
