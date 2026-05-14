"""Tests for v0.8.0 / Arch-1 (DRF-685): CandidateContext Protocol + generic ItemT."""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from ayla_ai_core.context import (
    CandidateContext,
    SpecialistCandidate,
    SpecialistContext,
    build_specialist_context_from_candidates,
)


class TestCandidateContextStructuralCheck:
    """v0.8.0 / Arch-1: CandidateContext is a runtime_checkable Protocol.
    Existing booking-domain SpecialistContext must satisfy it; arbitrary
    non-booking dataclasses with the same shape do too.
    """

    def test_specialist_context_satisfies_protocol(self) -> None:
        """The booking-domain class must already conform — no migration."""
        candidates = [
            SpecialistCandidate(id=1, name="Анна", specialization="м", services=[(10, "s")]),
        ]
        ctx = build_specialist_context_from_candidates(candidates, tenant_id="t")
        assert isinstance(ctx, CandidateContext)

    def test_non_booking_dataclass_with_same_shape_passes(self) -> None:
        """Arbitrary dataclass matching the Protocol fields passes
        isinstance — the whole point of widening from concrete-class
        isinstance to structural Protocol."""

        @dataclass(frozen=True)
        class KbChunk:
            id: str
            content: str

        @dataclass(frozen=True)
        class FAQContext:
            candidates: list[KbChunk]
            candidate_ids: frozenset[str]
            summary_text: str
            tenant_id: str

        faq_ctx = FAQContext(
            candidates=[KbChunk(id="chunk-1", content="text")],
            candidate_ids=frozenset({"chunk-1"}),
            summary_text="KB summary",
            tenant_id="some-tenant",
        )
        assert isinstance(faq_ctx, CandidateContext)

    def test_object_missing_tenant_id_fails(self) -> None:
        """Missing a required attribute → not a CandidateContext."""

        @dataclass
        class Incomplete:
            candidates: list
            candidate_ids: frozenset
            summary_text: str
            # tenant_id missing

        bad = Incomplete(
            candidates=[],
            candidate_ids=frozenset(),
            summary_text="",
        )
        assert not isinstance(bad, CandidateContext)

    def test_plain_string_is_not_a_candidate_context(self) -> None:
        """Sanity — a string doesn't have the required attributes."""
        assert not isinstance("not-a-context", CandidateContext)

    def test_simplenamespace_with_fields_passes(self) -> None:
        """SimpleNamespace duck-typed to the right fields satisfies Protocol —
        useful for inline mocks in third-party test suites."""
        ns_ctx = SimpleNamespace(
            candidates=[],
            candidate_ids=frozenset(),
            summary_text="",
            tenant_id="t",
        )
        assert isinstance(ns_ctx, CandidateContext)


class TestProtocolFlowsThroughOrchestrator:
    """Integration: when context_builder returns a custom CandidateContext
    impl that's NOT a SpecialistContext, the orchestrator's
    isinstance check passes (v0.8.0 Arch-1 widening) — pre-v0.8.0 it
    would have rejected with TypeError."""

    @pytest.mark.asyncio
    async def test_orchestrator_accepts_protocol_satisfying_custom_class(
        self,
    ) -> None:
        from ayla_ai_core.orchestrator import AIConcierge

        @dataclass(frozen=True)
        class CustomCtx:
            candidates: list
            candidate_ids: frozenset
            summary_text: str
            tenant_id: str
            # Booking-shape extras the bundled dispatch_tool_call uses.
            # Non-booking consumers would inject their own tool_dispatcher
            # and wouldn't need these, but the orchestrator still casts to
            # SpecialistContext for the default code path — so we provide them.
            candidate_service_ids: frozenset
            by_id: dict

        ctx = CustomCtx(
            candidates=[],
            candidate_ids=frozenset(),
            summary_text="",
            tenant_id="custom-tenant",
            candidate_service_ids=frozenset(),
            by_id={},
        )

        # Mock the openai_client + store + prompt_renderer so we can exercise
        # the context-validation gate without a real LLM round trip.
        class FakeStore:
            def resolve_active_conversation(self, _user_key):
                return SimpleNamespace(id=1)

            def save_message(self, conversation, **_kwargs):
                return SimpleNamespace(id=1)

            def load_recent_history(self, _conv, **_kwargs):
                return []

        class FakeClient:
            def __init__(self):
                self.chat = SimpleNamespace(completions=self)

            async def create(self, **_kwargs):
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content="ok", tool_calls=None),
                        ),
                    ],
                    usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
                )

        concierge = AIConcierge(
            openai_client=FakeClient(),
            store=FakeStore(),
            context_builder=lambda: ctx,
        )

        # Pre-v0.8.0 this would have raised TypeError("must return SpecialistContext").
        # v0.8.0 Arch-1: structural Protocol passes.
        result = await concierge.send_message(
            user_key=1,
            message_text="hi",
            prompt_renderer=lambda _: "system",
        )
        assert result.content == "ok"


class TestCandidateContextDocstringContract:
    """The Protocol's docstring promises certain fields. This test pins
    the field set so accidental additions/removals fail loud."""

    def test_required_attribute_names_are_stable(self) -> None:
        """If you add or remove a required attribute, update the docstring
        in context.py AND v0.9.0 migration notes — this test forces the
        conversation."""
        required = {"candidates", "candidate_ids", "summary_text", "tenant_id"}
        # Read the annotations on the Protocol class.
        annotations = set(CandidateContext.__annotations__.keys())
        assert annotations == required, (
            f"CandidateContext Protocol attribute set changed: "
            f"expected {required}, got {annotations}. "
            "Update docstring + CHANGELOG migration notes if intentional."
        )

    def test_specialist_context_remains_a_valid_alias_path(self) -> None:
        """v0.8.0 promises that SpecialistContext keeps working as the
        booking-domain concrete impl. This test pins that fact."""
        assert SpecialistContext is not None
        # SpecialistContext is a dataclass subscriptable for generic;
        # the runtime instance must still pass the Protocol check.
        ctx = build_specialist_context_from_candidates(
            [SpecialistCandidate(id=1, name="A", specialization="m", services=[(10, "s")])],
            tenant_id="t",
        )
        assert isinstance(ctx, SpecialistContext)
        assert isinstance(ctx, CandidateContext)


class TestArch4DeprecatedAliasesRemoved:
    """v0.8.0 (Arch-4 / DRF-688): MasterCandidate / MasterContext /
    build_master_context_from_candidates have been REMOVED. Pin this so
    a future accidental re-introduction fails CI loudly."""

    def test_master_candidate_alias_is_gone(self) -> None:
        import ayla_ai_core
        import ayla_ai_core.context as ctx_mod

        assert not hasattr(ctx_mod, "MasterCandidate")
        assert not hasattr(ayla_ai_core, "MasterCandidate")

    def test_master_context_alias_is_gone(self) -> None:
        import ayla_ai_core
        import ayla_ai_core.context as ctx_mod

        assert not hasattr(ctx_mod, "MasterContext")
        assert not hasattr(ayla_ai_core, "MasterContext")

    def test_build_master_context_alias_is_gone(self) -> None:
        import ayla_ai_core
        import ayla_ai_core.context as ctx_mod

        assert not hasattr(ctx_mod, "build_master_context_from_candidates")
        assert not hasattr(ayla_ai_core, "build_master_context_from_candidates")


class TestArch3DjangoExtra:
    """v0.8.0 (Arch-3 / DRF-687): Django moved from runtime deps to
    [django] optional extra. The library has no `import django` in src/.
    """

    def test_no_django_import_in_src(self) -> None:
        """Smoke: the library must not import django at import time so
        consumers without the [django] extra can use ayla."""
        import pathlib

        import ayla_ai_core

        src_root = pathlib.Path(ayla_ai_core.__file__).parent
        for py_file in src_root.rglob("*.py"):
            text = py_file.read_text(encoding="utf-8")
            assert "import django" not in text, (
                f"{py_file} contains `import django` — v0.8.0 promises no "
                "Django runtime dep."
            )
            assert "from django" not in text, (
                f"{py_file} contains `from django` — same as above."
            )
