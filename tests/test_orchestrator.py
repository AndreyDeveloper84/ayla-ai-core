"""Tests для AIConcierge — full pipeline с mocked OpenAI client + in-memory store.

Проверяемые invariants:
1. send_message returns ChatResponseDTO с правильными conversation_id / content
2. user message сохранён первым, assistant — после
3. tool_call в response → action_type / action_data заполнены
4. plain text response → action_type/data = None
5. context_builder вызывается; результат — обязан быть MasterContext
6. system_prompt инжектится в LLM messages первым
7. history включён в LLM messages (только user/assistant роли)
8. telemetry: tokens_in/out, latency_ms попадают в save_message
9. anti-hallucination: LLM выдал левый master_id → action_type=ask_clarification
10. async pipeline работает (sync_to_async wrappers корректны)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import pytest

from ayla_ai_core.context import MasterCandidate, build_master_context_from_candidates
from ayla_ai_core.orchestrator import (
    AIConcierge,
    ChatResponseDTO,
    MessageRole,
)
from ayla_ai_core.tools import ActionType

# ─── In-memory ConversationStore ──────────────────────────────────────────


@dataclass
class FakeConversation:
    id: UUID = field(default_factory=uuid4)
    is_active: bool = True
    messages: list = field(default_factory=list)


@dataclass
class FakeMessage:
    id: UUID = field(default_factory=uuid4)
    role: str = ""
    content: str = ""
    action_type: str = ""
    action_data: dict | None = None
    tool_call: dict | None = None
    tool_call_id: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: int | None = None


class FakeStore:
    """In-memory ConversationStore — Protocol-compatible через duck-typing."""

    def __init__(self) -> None:
        self.conversations: dict[Any, FakeConversation] = {}

    def resolve_active_conversation(self, user_key: Any) -> FakeConversation:
        if user_key not in self.conversations:
            self.conversations[user_key] = FakeConversation()
        return self.conversations[user_key]

    def save_message(self, conversation: FakeConversation, **kwargs) -> FakeMessage:
        msg = FakeMessage(**kwargs)
        conversation.messages.append(msg)
        return msg

    def load_recent_history(
        self, conversation: FakeConversation, *,
        exclude_id: Any = None, limit: int = 10,
    ) -> list[FakeMessage]:
        msgs = [m for m in conversation.messages if m.id != exclude_id]
        return msgs[-limit:]


# ─── Mock OpenAI client ───────────────────────────────────────────────────


class MockCompletion:
    def __init__(
        self,
        *,
        content: str = "",
        tool_call_name: str | None = None,
        tool_call_args: dict | None = None,
        tokens_in: int = 100,
        tokens_out: int = 50,
    ) -> None:
        tool_calls = None
        if tool_call_name is not None:
            tool_calls = [SimpleNamespace(
                id="tc_1",
                function=SimpleNamespace(
                    name=tool_call_name,
                    arguments=json.dumps(tool_call_args or {}),
                ),
            )]
        self.choices = [SimpleNamespace(
            message=SimpleNamespace(content=content, tool_calls=tool_calls),
        )]
        self.usage = SimpleNamespace(
            prompt_tokens=tokens_in,
            completion_tokens=tokens_out,
        )


class MockChatCompletions:
    def __init__(self, completion: MockCompletion) -> None:
        self._completion = completion
        self.last_request: dict | None = None

    async def create(self, **kwargs) -> MockCompletion:
        self.last_request = kwargs
        return self._completion


class MockOpenAIClient:
    def __init__(self, completion: MockCompletion) -> None:
        self.chat = SimpleNamespace(completions=MockChatCompletions(completion))


# ─── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def master_context():
    candidates = [
        MasterCandidate(id=1, name="Анна", specialization="массаж",
                        services=[(10, "массаж спины")]),
        MasterCandidate(id=2, name="Борис", specialization="спа",
                        services=[(11, "СПА")]),
    ]
    return build_master_context_from_candidates(candidates)


@pytest.fixture
def store():
    return FakeStore()


@pytest.fixture
def user_key():
    """User key — может быть любой immutable. Используем int для теста."""
    return 42


@pytest.fixture
def prompt_renderer(master_context):
    """Простой stub — игнорирует context, возвращает фиксированный prompt."""
    def _render(ctx) -> str:
        return f"You are Алина. Masters: {len(ctx.candidates)}"
    return _render


# ─── Tests ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_plain_text_response_no_tool_call(
    store, master_context, user_key, prompt_renderer
) -> None:
    """LLM ответил pure text — action_type/data = None."""
    client = MockOpenAIClient(MockCompletion(content="Привет, рада помочь."))
    concierge = AIConcierge(
        openai_client=client,
        store=store,
        context_builder=lambda: master_context,
    )

    result = await concierge.send_message(
        user_key=user_key,
        message_text="Привет",
        prompt_renderer=prompt_renderer,
    )

    assert isinstance(result, ChatResponseDTO)
    assert result.content == "Привет, рада помочь."
    assert result.action_type is None
    assert result.action_data is None


@pytest.mark.asyncio
async def test_tool_call_show_masters_returns_action(
    store, master_context, user_key, prompt_renderer
) -> None:
    """LLM вызвал show_masters с валидным master_id → action_type заполнен."""
    client = MockOpenAIClient(MockCompletion(
        tool_call_name="show_masters",
        tool_call_args={"master_ids": [1], "explanation": "Лучший выбор"},
    ))
    concierge = AIConcierge(
        openai_client=client, store=store,
        context_builder=lambda: master_context,
    )

    result = await concierge.send_message(
        user_key=user_key,
        message_text="Подбери массажиста",
        prompt_renderer=prompt_renderer,
    )

    assert result.action_type == ActionType.SHOW_MASTERS
    assert len(result.action_data["masters"]) == 1


@pytest.mark.asyncio
async def test_anti_hallucination_invalid_master_id_falls_to_clarification(
    store, master_context, user_key, prompt_renderer
) -> None:
    """LLM выдумал master_id=999 → handler фильтрует → ask_clarification."""
    client = MockOpenAIClient(MockCompletion(
        tool_call_name="show_masters",
        tool_call_args={"master_ids": [999], "explanation": "x"},
    ))
    concierge = AIConcierge(
        openai_client=client, store=store,
        context_builder=lambda: master_context,
    )

    result = await concierge.send_message(
        user_key=user_key,
        message_text="x",
        prompt_renderer=prompt_renderer,
    )

    # Anti-hallucination: вместо broken card — ask_clarification
    assert result.action_type == ActionType.ASK_CLARIFICATION


@pytest.mark.asyncio
async def test_user_message_saved_first_then_assistant(
    store, master_context, user_key, prompt_renderer
) -> None:
    client = MockOpenAIClient(MockCompletion(content="ок"))
    concierge = AIConcierge(
        openai_client=client, store=store,
        context_builder=lambda: master_context,
    )

    await concierge.send_message(
        user_key=user_key, message_text="Привет",
        prompt_renderer=prompt_renderer,
    )

    conv = store.conversations[user_key]
    assert len(conv.messages) == 2
    assert conv.messages[0].role == MessageRole.USER
    assert conv.messages[0].content == "Привет"
    assert conv.messages[1].role == MessageRole.ASSISTANT
    assert conv.messages[1].content == "ок"


@pytest.mark.asyncio
async def test_telemetry_tokens_and_latency_persisted(
    store, master_context, user_key, prompt_renderer
) -> None:
    client = MockOpenAIClient(MockCompletion(
        content="ок", tokens_in=123, tokens_out=45,
    ))
    concierge = AIConcierge(
        openai_client=client, store=store,
        context_builder=lambda: master_context,
    )

    await concierge.send_message(
        user_key=user_key, message_text="x",
        prompt_renderer=prompt_renderer,
    )

    conv = store.conversations[user_key]
    assistant_msg = conv.messages[1]
    assert assistant_msg.tokens_in == 123
    assert assistant_msg.tokens_out == 45
    assert assistant_msg.latency_ms is not None
    assert assistant_msg.latency_ms >= 0


@pytest.mark.asyncio
async def test_system_prompt_injected_first_in_llm_messages(
    store, master_context, user_key, prompt_renderer
) -> None:
    client = MockOpenAIClient(MockCompletion(content="ок"))
    concierge = AIConcierge(
        openai_client=client, store=store,
        context_builder=lambda: master_context,
    )

    await concierge.send_message(
        user_key=user_key, message_text="вопрос",
        prompt_renderer=prompt_renderer,
    )

    sent_messages = client.chat.completions.last_request["messages"]
    assert sent_messages[0]["role"] == "system"
    assert "Алина" in sent_messages[0]["content"]
    # Last сообщение — user
    assert sent_messages[-1]["role"] == "user"
    assert sent_messages[-1]["content"] == "вопрос"


@pytest.mark.asyncio
async def test_history_replayed_to_llm(
    store, master_context, user_key, prompt_renderer
) -> None:
    """Предыдущие сообщения попадают в LLM context (но не текущее user msg)."""
    client = MockOpenAIClient(MockCompletion(content="ок"))
    concierge = AIConcierge(
        openai_client=client, store=store,
        context_builder=lambda: master_context,
    )

    # Первый turn
    await concierge.send_message(
        user_key=user_key, message_text="первое",
        prompt_renderer=prompt_renderer,
    )
    # Второй turn — должны увидеть первое в LLM messages
    await concierge.send_message(
        user_key=user_key, message_text="второе",
        prompt_renderer=prompt_renderer,
    )

    sent = client.chat.completions.last_request["messages"]
    # system + 2 history (user "первое", assistant "ок") + 1 new user "второе" = 4
    user_messages = [m for m in sent if m["role"] == "user"]
    assert any(m["content"] == "первое" for m in user_messages)
    assert any(m["content"] == "второе" for m in user_messages)


@pytest.mark.asyncio
async def test_history_limit_respected(
    store, master_context, user_key, prompt_renderer
) -> None:
    """history_limit=2 → в LLM messages только последние 2 history msgs (после exclude)."""
    client = MockOpenAIClient(MockCompletion(content="ок"))
    concierge = AIConcierge(
        openai_client=client, store=store,
        context_builder=lambda: master_context,
        history_limit=2,
    )

    # 3 turn-а — после третьего история = msgs[1..6], последние 2 после exclude = 2
    for i in range(3):
        await concierge.send_message(
            user_key=user_key, message_text=f"msg-{i}",
            prompt_renderer=prompt_renderer,
        )

    sent = client.chat.completions.last_request["messages"]
    # system + 2 history + 1 new user = 4 maximum
    assert len(sent) <= 4


@pytest.mark.asyncio
async def test_context_builder_must_return_master_context(
    store, user_key, prompt_renderer
) -> None:
    """Defensive: builder вернул не MasterContext → TypeError."""
    client = MockOpenAIClient(MockCompletion(content="ок"))
    concierge = AIConcierge(
        openai_client=client, store=store,
        context_builder=lambda: "not-a-context",
    )

    with pytest.raises(TypeError, match="MasterContext"):
        await concierge.send_message(
            user_key=user_key, message_text="x",
            prompt_renderer=prompt_renderer,
        )


@pytest.mark.asyncio
async def test_async_context_builder_supported(
    store, master_context, user_key, prompt_renderer
) -> None:
    """context_builder может быть async (для thread-pooled ORM)."""
    async def async_builder():
        return master_context

    client = MockOpenAIClient(MockCompletion(content="ок"))
    concierge = AIConcierge(
        openai_client=client, store=store,
        context_builder=async_builder,
    )

    result = await concierge.send_message(
        user_key=user_key, message_text="x",
        prompt_renderer=prompt_renderer,
    )
    assert result.content == "ок"


@pytest.mark.asyncio
async def test_sync_context_builder_runs_off_event_loop(
    store, master_context, user_key, prompt_renderer
) -> None:
    """Sync builder (типичный case бота — Django ORM) — должен быть обёрнут
    в sync_to_async, иначе блокирующий I/O сидит на event loop'е и Django
    кидает SynchronousOnlyOperation.

    Test strategy: builder засекает thread_id; main event loop — другой thread.
    """
    import threading

    main_thread_id = threading.get_ident()
    builder_thread_id = None

    def sync_builder():
        nonlocal builder_thread_id
        builder_thread_id = threading.get_ident()
        return master_context

    client = MockOpenAIClient(MockCompletion(content="ок"))
    concierge = AIConcierge(
        openai_client=client, store=store,
        context_builder=sync_builder,
    )
    await concierge.send_message(
        user_key=user_key, message_text="x",
        prompt_renderer=prompt_renderer,
    )

    # Sync builder через sync_to_async выполнен в thread-pool, не на main event loop
    assert builder_thread_id is not None
    assert builder_thread_id != main_thread_id


@pytest.mark.asyncio
async def test_async_coroutine_context_builder_not_double_wrapped(
    store, master_context, user_key, prompt_renderer
) -> None:
    """Async coroutine-функция должна вызываться напрямую (await), без sync_to_async wrapper."""
    call_count = 0

    async def async_builder():
        nonlocal call_count
        call_count += 1
        return master_context

    client = MockOpenAIClient(MockCompletion(content="ок"))
    concierge = AIConcierge(
        openai_client=client, store=store,
        context_builder=async_builder,
    )
    await concierge.send_message(
        user_key=user_key, message_text="x",
        prompt_renderer=prompt_renderer,
    )
    # Builder вызван ровно раз (не дважды через sync_to_async + await)
    assert call_count == 1


@pytest.mark.asyncio
async def test_tool_call_raw_persisted_for_audit(
    store, master_context, user_key, prompt_renderer
) -> None:
    """raw OpenAI tool_call (id, name, arguments string) сохраняется в Message.tool_call."""
    client = MockOpenAIClient(MockCompletion(
        tool_call_name="show_masters",
        tool_call_args={"master_ids": [1], "explanation": "x"},
    ))
    concierge = AIConcierge(
        openai_client=client, store=store,
        context_builder=lambda: master_context,
    )

    await concierge.send_message(
        user_key=user_key, message_text="x",
        prompt_renderer=prompt_renderer,
    )

    assistant_msg = store.conversations[user_key].messages[1]
    assert assistant_msg.tool_call is not None
    assert assistant_msg.tool_call["name"] == "show_masters"
    assert assistant_msg.tool_call_id == "tc_1"


@pytest.mark.asyncio
async def test_tools_passed_to_openai_create(
    store, master_context, user_key, prompt_renderer
) -> None:
    """TOOL_DEFINITIONS передаются в LLM call."""
    client = MockOpenAIClient(MockCompletion(content="ок"))
    concierge = AIConcierge(
        openai_client=client, store=store,
        context_builder=lambda: master_context,
    )

    await concierge.send_message(
        user_key=user_key, message_text="x",
        prompt_renderer=prompt_renderer,
    )

    tools = client.chat.completions.last_request["tools"]
    assert len(tools) == 5  # все 5 standard tools
    names = {t["function"]["name"] for t in tools}
    assert names == {"show_masters", "show_slots", "confirm_booking",
                     "show_my_bookings", "ask_clarification"}


@pytest.mark.asyncio
async def test_one_active_conversation_per_user_key(
    store, master_context, user_key, prompt_renderer
) -> None:
    """Множественные send_message в один user_key → одна Conversation."""
    client = MockOpenAIClient(MockCompletion(content="ок"))
    concierge = AIConcierge(
        openai_client=client, store=store,
        context_builder=lambda: master_context,
    )

    r1 = await concierge.send_message(
        user_key=user_key, message_text="первое",
        prompt_renderer=prompt_renderer,
    )
    r2 = await concierge.send_message(
        user_key=user_key, message_text="второе",
        prompt_renderer=prompt_renderer,
    )

    assert r1.conversation_id == r2.conversation_id
    assert len(store.conversations) == 1


@pytest.mark.asyncio
async def test_different_user_keys_get_separate_conversations(
    store, master_context, prompt_renderer
) -> None:
    client = MockOpenAIClient(MockCompletion(content="ок"))
    concierge = AIConcierge(
        openai_client=client, store=store,
        context_builder=lambda: master_context,
    )

    r1 = await concierge.send_message(
        user_key=1, message_text="x", prompt_renderer=prompt_renderer,
    )
    r2 = await concierge.send_message(
        user_key=2, message_text="x", prompt_renderer=prompt_renderer,
    )

    assert r1.conversation_id != r2.conversation_id
    assert len(store.conversations) == 2
