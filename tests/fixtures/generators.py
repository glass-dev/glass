"""Handling of array backends."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import glass.rng

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

    from glass._types import UnifiedGenerator



@pytest.fixture(scope="session")
def get_rng() -> Callable[..., UnifiedGenerator]:
    """
    Return an RNG fixture for non array API tests.

    Use `urng` for array API tests.
    """
    return lambda xp : glass.rng.default_rng(xp=xp)


@pytest.fixture
def urng(xp: ModuleType) -> UnifiedGenerator:
    """
    Fixture for a unified RNG interface.

    Access the relevant RNG using `urng.` in tests.

    Must be used with the `xp` fixture. Use `rng` for non array API tests.

    """
    return glass.rng.default_rng(xp=xp)
