"""Smoke tests — package installs, version is readable.

Will be expanded in DRF-237..239 as AIConcierge / tools / context get extracted.
"""
from __future__ import annotations

import ayla_ai_core


def test_package_imports() -> None:
    assert ayla_ai_core.__version__ == "0.1.0"


def test_package_has_version_string() -> None:
    assert isinstance(ayla_ai_core.__version__, str)
    assert len(ayla_ai_core.__version__) > 0
