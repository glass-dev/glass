from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from glass import _rng

if TYPE_CHECKING:
    from types import ModuleType

    from glass._types import UnifiedGenerator


@pytest.fixture
def xp(xpb: ModuleType) -> ModuleType:
    """Alias of Fixture for array backend to be used in regression tests."""
    return xpb


@pytest.fixture
def urngb(xpb: ModuleType) -> UnifiedGenerator:
    """
    Fixture for a unified RNG interface to be used in regression tests.

    Access the relevant RNG using `urngb.` in tests.

    Must be used with the `xpb` fixture.

    """
    return _rng.rng_dispatcher(xp=xpb)
