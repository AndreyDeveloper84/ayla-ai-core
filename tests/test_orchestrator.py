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
    # v0.7.0: tenant_id mandatory. Synthetic test value — clearly not real.
    return build_master_context_from_candidates(candidates, tenant_id="test-tenant")


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

    with pytest.raises(TypeError, match="SpecialistContext"):
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
async def test_orchestrator_with_uuid_id_parser_end_to_end(
    store, prompt_renderer
) -> None:
    """Ayla scenario: AIConcierge с _safe_uuid + SpecialistContext[UUID]."""
    from uuid import UUID

    from ayla_ai_core import _safe_uuid
    from ayla_ai_core.context import (
        SpecialistCandidate,
        build_specialist_context_from_candidates,
    )

    uid_anna = UUID("11111111-1111-1111-1111-111111111111")
    sid_back = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    candidates = [
        SpecialistCandidate(
            id=uid_anna, name="Анна", specialization="массаж",
            services=[(sid_back, "массаж спины")],
        ),
    ]
    uuid_context = build_specialist_context_from_candidates(candidates, tenant_id="ayla")

    # LLM emits UUID-string
    client = MockOpenAIClient(MockCompletion(
        tool_call_name="show_masters",
        tool_call_args={"master_ids": ["11111111-1111-1111-1111-111111111111"], "explanation": "x"},
    ))
    concierge = AIConcierge(
        openai_client=client,
        store=store,
        context_builder=lambda: uuid_context,
        id_parser=_safe_uuid,
    )

    result = await concierge.send_message(
        user_key="ayla-user-uuid",
        message_text="Подбери массажиста",
        prompt_renderer=prompt_renderer,
    )

    assert result.action_type == ActionType.SHOW_MASTERS
    assert len(result.action_data["masters"]) == 1
    assert result.action_data["masters"][0]["master"]["name"] == "Анна"


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


# ─── DRF-241 / 0.6.0 — custom tool_dispatcher DI hook ─────────────────────


@pytest.mark.asyncio
async def test_custom_tool_dispatcher_called_when_provided(
    store, master_context, user_key, prompt_renderer
) -> None:
    """When tool_dispatcher is injected, AIConcierge calls it instead of
    the bundled dispatch_tool_call. Consumer is responsible for parsing
    the tool_call (name, arguments) however its local wire-format wants."""
    from ayla_ai_core.tool_handlers import ToolResult

    captured: dict = {}

    def custom_dispatcher(tool_call, context):
        captured["name"] = tool_call.function.name
        captured["args"] = tool_call.function.arguments
        captured["candidates_seen"] = len(context.candidates)
        # Consumer's own wire-format — Ayla returns "show_specialists"
        # not "show_masters".
        return ToolResult(
            action_type="show_specialists",
            action_data={"specialist_ids": ["abc-123"]},
        )

    client = MockOpenAIClient(MockCompletion(
        tool_call_name="show_specialists",
        tool_call_args={"specialist_ids": ["abc-123"], "explanation": "x"},
    ))
    concierge = AIConcierge(
        openai_client=client, store=store,
        context_builder=lambda: master_context,
        tool_dispatcher=custom_dispatcher,
    )

    result = await concierge.send_message(
        user_key=user_key, message_text="хочу маникюр",
        prompt_renderer=prompt_renderer,
    )

    assert captured["name"] == "show_specialists"
    assert "specialist_ids" in captured["args"]
    assert captured["candidates_seen"] == 2
    assert result.action_type == "show_specialists"
    assert result.action_data == {"specialist_ids": ["abc-123"]}


@pytest.mark.asyncio
async def test_default_dispatcher_used_when_tool_dispatcher_is_none(
    store, master_context, user_key, prompt_renderer
) -> None:
    """Backward-compat: tool_dispatcher=None (default) keeps the bundled
    dispatch_tool_call path. Every existing consumer keeps current behaviour."""
    client = MockOpenAIClient(MockCompletion(
        tool_call_name="show_masters",
        tool_call_args={"master_ids": [1], "explanation": "ok"},
    ))
    concierge = AIConcierge(
        openai_client=client, store=store,
        context_builder=lambda: master_context,
        # tool_dispatcher omitted entirely → defaults to None
    )

    result = await concierge.send_message(
        user_key=user_key, message_text="x",
        prompt_renderer=prompt_renderer,
    )

    # Bundled handler renders "show_masters" via ActionType.SHOW_MASTERS.
    assert result.action_type == ActionType.SHOW_MASTERS


@pytest.mark.asyncio
async def test_custom_dispatcher_receives_tool_call_object_shape(
    store, master_context, user_key, prompt_renderer
) -> None:
    """The dispatcher gets the raw OpenAI tool_call object — same shape the
    bundled dispatch_tool_call expects (`.function.name`, `.function.arguments`).
    Consumers parse JSON args themselves (they own the wire-format)."""
    from ayla_ai_core.tool_handlers import ToolResult

    seen_id = []

    def dispatcher(tc, ctx):
        seen_id.append(getattr(tc, "id", None))
        return ToolResult(action_type="custom", action_data=None)

    client = MockOpenAIClient(MockCompletion(
        tool_call_name="anything",
        tool_call_args={},
    ))
    concierge = AIConcierge(
        openai_client=client, store=store,
        context_builder=lambda: master_context,
        tool_dispatcher=dispatcher,
    )

    await concierge.send_message(
        user_key=user_key, message_text="x",
        prompt_renderer=prompt_renderer,
    )

    # MockCompletion sets tool_call.id = "tc_1"; the dispatcher saw it.
    assert seen_id == ["tc_1"]


@pytest.mark.asyncio
async def test_custom_dispatcher_skips_id_parser_and_resolvers(
    store, master_context, user_key, prompt_renderer
) -> None:
    """When a custom dispatcher is provided, id_parser / master_resolver /
    service_resolver are NOT applied — the consumer owns validation. This
    matches Ayla's pattern: local handlers do their own anti-hallucination."""
    from ayla_ai_core.tool_handlers import ToolResult

    parser_calls = []

    def tracking_parser(value):
        parser_calls.append(value)
        return value

    def custom_dispatcher(tool_call, context):
        return ToolResult(action_type="show_specialists", action_data={})

    client = MockOpenAIClient(MockCompletion(
        tool_call_name="show_specialists",
        tool_call_args={"specialist_ids": ["abc"]},
    ))
    concierge = AIConcierge(
        openai_client=client, store=store,
        context_builder=lambda: master_context,
        id_parser=tracking_parser,
        tool_dispatcher=custom_dispatcher,
    )

    await concierge.send_message(
        user_key=user_key, message_text="x",
        prompt_renderer=prompt_renderer,
    )

    # id_parser was passed but never invoked — custom dispatcher handles
    # all validation including ID parsing.
    assert parser_calls == []


# ─── B1 regression tests (v0.7.0) ─────────────────────────────────────────


class TestComposeMessages:
    """B1 (v0.7.0): _compose_messages must skip assistant turns whose
    content is empty AND tool_calls is empty.

    OpenAI rejects such messages with HTTP 400 after the first
    tool_call lands in conversation history. Assistant turns that
    DO carry tool_calls are preserved — those messages encode the
    function-call pairing the API needs.
    """

    @staticmethod
    def _compose(history, *, system_prompt="sys", user_text="hi"):
        # Import locally so we don't reshape the module-level test imports.
        from ayla_ai_core.orchestrator import _compose_messages

        return _compose_messages(
            system_prompt=system_prompt, history=history, user_text=user_text
        )

    def test_skips_empty_assistant_turns_without_tool_calls(self):
        """History [user, assistant("", no tool_calls), user] → 4 msgs
        (system + user + user + new user), not 5 — the empty assistant
        turn is dropped."""
        history = [
            SimpleNamespace(role=MessageRole.USER, content="prev question"),
            SimpleNamespace(role=MessageRole.ASSISTANT, content="", tool_calls=None),
            SimpleNamespace(role=MessageRole.USER, content="follow-up"),
        ]
        msgs = self._compose(history)
        assert len(msgs) == 4
        assert msgs[0]["role"] == "system"
        assert [m["role"] for m in msgs[1:]] == [
            MessageRole.USER,
            MessageRole.USER,
            MessageRole.USER,
        ]
        # The two history user contents survive, plus the new user_text.
        contents = [m["content"] for m in msgs[1:]]
        assert contents == ["prev question", "follow-up", "hi"]

    def test_keeps_assistant_with_tool_calls_even_if_content_empty(self):
        """Assistant turns carrying tool_calls are preserved even when
        content is empty — OpenAI needs the call/response pairing."""
        tool_call_stub = SimpleNamespace(id="tc_1", function=SimpleNamespace())
        history = [
            SimpleNamespace(role=MessageRole.USER, content="book me a slot"),
            SimpleNamespace(
                role=MessageRole.ASSISTANT, content="", tool_calls=[tool_call_stub]
            ),
        ]
        msgs = self._compose(history)
        # system + user + assistant("") + new user
        assert len(msgs) == 4
        assert msgs[2]["role"] == MessageRole.ASSISTANT
        assert msgs[2]["content"] == ""

    def test_keeps_assistant_with_content_even_if_no_tool_calls(self):
        """Normal assistant text turns are unaffected."""
        history = [
            SimpleNamespace(role=MessageRole.USER, content="q"),
            SimpleNamespace(
                role=MessageRole.ASSISTANT, content="my answer", tool_calls=None
            ),
        ]
        msgs = self._compose(history)
        assert len(msgs) == 4
        assert msgs[2]["content"] == "my answer"

    def test_filters_non_user_assistant_roles(self):
        """system/tool roles in history are dropped — pre-existing behaviour."""
        history = [
            SimpleNamespace(role="system", content="old prompt"),
            SimpleNamespace(role=MessageRole.USER, content="hello"),
            SimpleNamespace(role="tool", content="result"),
        ]
        msgs = self._compose(history)
        # system + user(hello) + new user(hi)
        assert len(msgs) == 3


# ─── DRF-678 (v0.7.2 Perf-2): parallel tool_calls support ─────────────────


def _fake_completion_multi(tool_calls_specs: list[tuple[str, dict]]) -> SimpleNamespace:
    """Build a completion stub carrying multiple tool_calls.

    Each entry of ``tool_calls_specs`` is ``(name, args_dict)``; tool_call id
    is auto-numbered ``tc_0``, ``tc_1``, ... so tests can assert which
    primary call won.
    """
    tool_calls = [
        SimpleNamespace(
            id=f"tc_{i}",
            function=SimpleNamespace(name=name, arguments=json.dumps(args)),
        )
        for i, (name, args) in enumerate(tool_calls_specs)
    ]
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="", tool_calls=tool_calls),
            ),
        ],
    )


class TestParallelToolCalls:
    """v0.7.2 (DRF-678): _parse_completion must dispatch every tool_call,
    not just ``[0]``. gpt-4o emits 2-3 parallel calls with parallel_tool_calls=True
    by default, and v0.7.0 silently dropped tail items (real prod data loss).

    Merge strategy: "first non-clarification wins" for the primary; all
    remaining results return as ``extra_actions`` in original order.
    """

    @staticmethod
    def _parse(master_context, tool_calls_specs):
        from ayla_ai_core.orchestrator import _parse_completion
        from ayla_ai_core.tool_handlers import _safe_int

        return _parse_completion(
            _fake_completion_multi(tool_calls_specs),
            master_context,
            master_resolver=None,
            service_resolver=None,
            id_parser=_safe_int,
        )

    def test_single_tool_call_extra_actions_is_none(self, master_context):
        """Backward-compat: 1 tool_call -> extra_actions == None (v0.7.0 shape)."""
        _content, raw, action_type, _action_data, extras = self._parse(
            master_context,
            [("show_masters", {"master_ids": [1], "explanation": "Лучший"})],
        )
        assert action_type == "show_masters"
        assert extras is None
        assert raw["id"] == "tc_0"

    def test_double_tool_call_primary_first_non_clarification(self, master_context):
        """2 calls, both concrete -> primary == [0], extras == [1]."""
        _content, raw, action_type, _action_data, extras = self._parse(
            master_context,
            [
                ("show_masters", {"master_ids": [1], "explanation": "Сначала покажу мастеров"}),
                ("show_slots", {"master_id": 1, "service_id": 10, "date": "2026-05-20"}),
            ],
        )
        assert action_type == "show_masters"
        assert raw["id"] == "tc_0"
        assert extras == [
            {
                "action_type": "show_slots",
                "action_data": {"master_id": 1, "service_id": 10, "date": "2026-05-20"},
            },
        ]

    def test_triple_tool_call_carries_all_extras_in_order(self, master_context):
        """3 calls -> primary first non-clarification, 2 extras in order."""
        _content, _raw, action_type, _action_data, extras = self._parse(
            master_context,
            [
                ("show_masters", {"master_ids": [1], "explanation": "Шаг 1"}),
                ("show_slots", {"master_id": 1, "service_id": 10, "date": "2026-05-21"}),
                ("show_my_bookings", {"filter": "upcoming"}),
            ],
        )
        assert action_type == "show_masters"
        assert extras is not None
        assert len(extras) == 2
        assert [e["action_type"] for e in extras] == ["show_slots", "show_my_bookings"]

    def test_clarification_first_promotes_next_non_clarification_to_primary(
        self, master_context,
    ):
        """v0.7.2 merge rule: clarification at [0], concrete action at [1]
        -> primary becomes the concrete action; clarification carried in
        extras.

        Real-world trigger: gpt-4o sometimes emits an ask_clarification
        alongside a show_masters when it's unsure but still has candidates.
        Prior to v0.7.2 the clarification would silently shadow the
        useful action because we only read [0]."""
        _content, raw, action_type, _action_data, extras = self._parse(
            master_context,
            [
                ("ask_clarification", {"question": "Какой день удобнее?"}),
                ("show_masters", {"master_ids": [1], "explanation": "Пока вот мастер"}),
            ],
        )
        assert action_type == "show_masters"
        assert raw["id"] == "tc_1"  # primary's raw is the show_masters call
        assert extras is not None
        assert [e["action_type"] for e in extras] == ["ask_clarification"]

    def test_all_clarifications_falls_back_to_first(self, master_context):
        """No non-clarification result -> first clarification is the primary."""
        _content, raw, action_type, _action_data, extras = self._parse(
            master_context,
            [
                ("ask_clarification", {"question": "Когда?"}),
                ("ask_clarification", {"question": "С кем?"}),
            ],
        )
        assert action_type == "ask_clarification"
        assert raw["id"] == "tc_0"
        assert extras == [
            {
                "action_type": "ask_clarification",
                "action_data": {"question": "С кем?", "options": []},
            },
        ]

    def test_hallucinated_id_in_middle_drops_to_clarification_extra(
        self, master_context,
    ):
        """Tool call with hallucinated id -> ASK_CLARIFICATION; sibling
        valid calls still come through as primary + extras."""
        _content, _raw, action_type, _action_data, extras = self._parse(
            master_context,
            [
                ("show_slots", {"master_id": 999, "service_id": 10, "date": "2026-05-20"}),
                ("show_masters", {"master_ids": [1], "explanation": "Take this one"}),
            ],
        )
        # First call hallucinated -> fallback to ask_clarification (handlers
        # are side-effect-free, the clarification just rides along in extras).
        # Second call wins because it's concrete.
        assert action_type == "show_masters"
        assert extras is not None
        assert len(extras) == 1
        assert extras[0]["action_type"] == "ask_clarification"
