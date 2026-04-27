"""AIConcierge — главный chat-pipeline orchestrator.

Адаптация из `mysite/maxbot/ai_concierge.py` (а тот, в свою очередь, был портом
из `Ayla/djangoproject/ai/application/services/chat_service.py`). Извлечение
выполнено в DRF-237.

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

import inspect
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from asgiref.sync import sync_to_async

from ayla_ai_core.context import MasterContext
from ayla_ai_core.tool_handlers import (
    MasterResolver,
    ServiceResolver,
    dispatch_tool_call,
)
from ayla_ai_core.tools import TOOL_DEFINITIONS

__all__ = [
    "DEFAULT_HISTORY_LIMIT",
    "DEFAULT_MODEL_NAME",
    "AIConcierge",
    "ChatResponseDTO",
    "ConversationStore",
    "MessageRole",
]


logger = logging.getLogger("ayla_ai_core.orchestrator")

DEFAULT_HISTORY_LIMIT = 10
DEFAULT_MODEL_NAME = "gpt-4o-mini"


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
    action_type / action_data — tool action для UI рендера. None если LLM
    ответил pure-text (small-talk).
    """

    conversation_id: Any  # UUID для бота / Ayla, может быть int в legacy
    content: str
    action_type: str | None
    action_data: dict[str, Any] | None


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
) -> list[dict[str, Any]]:
    """OpenAI-format messages: system + history + new user.

    history elements должны иметь .role и .content атрибуты (duck-typed).
    Tool/system messages в истории пропускаются — на текущей итерации
    multi-step tool-use не делается.
    """
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    for h in history:
        h_role = getattr(h, "role", None)
        if h_role not in (MessageRole.USER, MessageRole.ASSISTANT):
            continue
        messages.append({"role": h_role, "content": getattr(h, "content", "") or ""})
    messages.append({"role": MessageRole.USER, "content": user_text})
    return messages


def _parse_completion(
    completion: Any,
    master_context: MasterContext,
    *,
    master_resolver: MasterResolver | None,
    service_resolver: ServiceResolver | None,
) -> tuple[str, dict | None, str | None, dict | None]:
    """Returns (content, tool_call_raw, action_type, action_data)."""
    msg = completion.choices[0].message
    tool_calls = getattr(msg, "tool_calls", None)
    content = msg.content or ""

    if not tool_calls:
        return content, None, None, None

    # Берём первый tool_call — наш flow одношаговый (LLM либо текст,
    # либо один tool-call, не цепочка)
    tc = tool_calls[0]
    result = dispatch_tool_call(
        tc, master_context,
        master_resolver=master_resolver,
        service_resolver=service_resolver,
    )

    raw = {
        "id": getattr(tc, "id", ""),
        "name": getattr(tc.function, "name", ""),
        "arguments": getattr(tc.function, "arguments", ""),
    }
    return content, raw, result.action_type, result.action_data


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
        tool_definitions: list[dict] | None = None,
    ) -> None:
        """Construct AIConcierge.

        openai_client: AsyncOpenAI — async OpenAI client (или совместимый).
        store: реализация ConversationStore (sync методы; будут обёрнуты в
            sync_to_async автоматически).
        context_builder: callable() → MasterContext (sync OR async). Должен
            быть дёшевым — вызывается на каждый turn.
        model_name: OpenAI model id. Default gpt-4o-mini (production-tested
            в боте Формулы).
        history_limit: max messages из истории для LLM context. Default 10.
        tool_definitions: override TOOL_DEFINITIONS. Default — все 5 standard.
        """
        self._openai_client = openai_client
        self._store = store
        self._context_builder = context_builder
        self._model_name = model_name
        self._history_limit = history_limit
        self._tool_definitions = tool_definitions or TOOL_DEFINITIONS

        # async-обёртки для sync ORM-методов store
        self._resolve_conv = sync_to_async(store.resolve_active_conversation)
        self._save_message = sync_to_async(store.save_message)
        self._load_history = sync_to_async(store.load_recent_history)

    async def send_message(
        self,
        *,
        user_key: Any,
        message_text: str,
        prompt_renderer: Callable[[MasterContext], str],
        master_resolver: MasterResolver | None = None,
        service_resolver: ServiceResolver | None = None,
    ) -> ChatResponseDTO:
        """Один turn AI Concierge: user message → assistant response с opt action.

        user_key: key для store.resolve_active_conversation (обычно BotUser/User
            instance, ID — зависит от store impl).
        message_text: текст от пользователя.
        prompt_renderer: callable(master_context) → system_prompt string.
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
        master_context_raw = self._context_builder()
        master_context = await _maybe_await(master_context_raw)
        if not isinstance(master_context, MasterContext):
            raise TypeError(
                f"context_builder must return MasterContext, got {type(master_context)}"
            )

        # 4. Load recent history
        history = await self._load_history(
            conversation,
            exclude_id=getattr(user_msg, "id", None),
            limit=self._history_limit,
        )

        # 5. Render system prompt
        system_prompt = prompt_renderer(master_context)

        # 6. Compose for OpenAI
        llm_messages = _compose_messages(
            system_prompt=system_prompt,
            history=history,
            user_text=message_text,
        )

        # 7. Call OpenAI с tools
        started = time.monotonic()
        completion = await self._openai_client.chat.completions.create(
            model=self._model_name,
            messages=llm_messages,
            tools=self._tool_definitions,
        )
        latency_ms = int((time.monotonic() - started) * 1000)

        # 8. Parse: content + opt action
        content, tool_call_raw, action_type, action_data = _parse_completion(
            completion, master_context,
            master_resolver=master_resolver,
            service_resolver=service_resolver,
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
        )

        return ChatResponseDTO(
            conversation_id=getattr(conversation, "id", None),
            content=content,
            action_type=action_type,
            action_data=action_data,
        )
