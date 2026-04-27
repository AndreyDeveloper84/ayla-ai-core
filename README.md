# ayla-ai-core

> Shared AI orchestration core для Ayla и бот Формулы тела. Извлечён из `mysite/maxbot/` (production-tested 30+ days) согласно `docs/BOT_CODE_AUDIT_2026-04.md` Variant C+ strategy.

**Status:** 🟡 0.1.0 — boilerplate только. Реальное содержимое появляется в DRF-237..239.

## Quick start

### Installation (editable, для разработки)

```bash
# В venv проекта-потребителя (ayla/djangoproject или mysite)
pip install -e ../ayla-ai-core
```

Зависимости (после полной экстракции):
- Python ≥3.12
- openai ≥1.40
- django ≥5.0
- asgiref, pydantic

### Verification

```python
import ayla_ai_core
print(ayla_ai_core.__version__)  # 0.1.0
```

## Roadmap (Phase A extraction)

| # | Module | Source (бот) | Linear |
|---|--------|--------------|--------|
| 1 | `orchestrator.AIConcierge` | `mysite/maxbot/ai_concierge.py` | DRF-237 |
| 2 | `tools.TOOL_DEFINITIONS` | `mysite/maxbot/ai_tools.py` | DRF-237 |
| 3 | `tool_handlers.dispatch_tool_call` | `mysite/maxbot/ai_tool_handlers.py` | DRF-237 |
| 4 | `context.SpecialistContext` | `mysite/maxbot/ai_context.py` | DRF-238 |
| 5 | `prompts.render_system_prompt` + `BrandVoiceConfig` | `mysite/maxbot/ai_prompts.py` | DRF-239 |

## Multi-tenant brand voices

Package supports multiple brand voices через `BrandVoiceConfig`:

- `FORMULA_TELA_VOICE` — для бота Формулы тела (legacy)
- `AYLA_MARKETPLACE_VOICE` — для Ayla mobile (новый)

Adding new voice = создать config dataclass, не форкать prompts.

## Development

```bash
# Установка dev-зависимостей
pip install -e ".[dev]"

# Тесты
pytest

# Lint
ruff check .

# Type check
mypy src/
```

## Versioning

Semver. Bot и Ayla pin'ятся на минорную версию (`0.1.x`) — minor releases совместимы внутри major.

## License

Proprietary. Internal use only by Ayla / Formula tela team.

## Связанные документы

- `ayla/djangoproject/docs/BOT_CODE_AUDIT_2026-04.md` — full audit + reuse map
- `ayla/djangoproject/docs/PRODUCT_AUDIT_2026-04.md` — strategic context
- Linear `Ayla` project — DRF-236..245 для Phase A+B work
