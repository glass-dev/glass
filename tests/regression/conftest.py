from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType


@pytest.fixture
def xp(xpb: ModuleType) -> ModuleType:
    """Alias of Fixture for array backend to be used in regression tests."""
    return xpb
