"""ayla-ai-core — shared AI orchestration core for Ayla and Formula tela bot.

Public API will be exposed here as modules are extracted from `mysite/maxbot/`
in DRF-237..239. Initial 0.1.0 release ships with version stub only — both
consumer projects (`ayla/djangoproject` and `mysite`) install editable to
verify wiring.

See `docs/BOT_CODE_AUDIT_2026-04.md` (in djangoproject) for extraction plan.
"""
from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["__version__"]
