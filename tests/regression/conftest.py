from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import glass.rng

if TYPE_CHECKING:
    from types import ModuleType

    from glass._types import UnifiedGenerator


@pytest.fixture
def xp(xpb: ModuleType) -> ModuleType:
    """Alias of Fixture for array backend to be used in regression tests."""
    return xpb


@pytest.fixture
def urng(xpb: ModuleType) -> UnifiedGenerator:
    """
    Fixture for a unified RNG interface to be used in regression tests.

    Access the relevant RNG using `urng.` in tests.

    Must be used with the `xp` fixture.

    """
    return glass.rng.default_rng(xp=xpb)
