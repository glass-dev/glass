"""Handling of array backends."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from glass import _rng

if TYPE_CHECKING:
    from types import ModuleType

    from glass._types import UnifiedGenerator


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    """
    RNG fixture for non array API tests.

    Use `urng` for array API tests.

    """
    return np.random.default_rng(seed=_rng.SEED)


@pytest.fixture
def urng(xp: ModuleType) -> UnifiedGenerator:
    """
    Fixture for a unified RNG interface.

    Access the relevant RNG using `urng.` in tests.

    Must be used with the `xp` fixture. Use `rng` for non array API tests.

    """
    return _rng.rng_dispatcher(xp=xp)


@pytest.fixture
def urngb(xpb: ModuleType) -> UnifiedGenerator:
    """
    Fixture for a unified RNG interface to be used in benchmarks.

    Access the relevant RNG using `urngb.` in tests.

    Must be used with the `xpb` fixture.

    """
    return _rng.rng_dispatcher(xp=xpb)
