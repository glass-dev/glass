"""Handling of array backends."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import numpy as np
import pytest

import glass._array_api_utils as _utils

with contextlib.suppress(ImportError):
    # only import if jax is available
    import glass.jax

if TYPE_CHECKING:
    from types import ModuleType

    from glass._types import UnifiedGenerator

SEED = 42


def _select_urng(xp: ModuleType) -> UnifiedGenerator:
    """Given an array backend `xp`, returns the matching rng."""
    if xp.__name__ == "jax.numpy":
        return glass.jax.Generator(seed=SEED)
    if xp.__name__ == "numpy":
        return np.random.default_rng(seed=SEED)
    if xp.__name__ == "array_api_strict":
        return _utils.Generator(seed=SEED)
    msg = "the array backend in not supported"
    raise NotImplementedError(msg)


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    """
    RNG fixture for non array API tests.

    Use `urng` for array API tests.
    """
    return np.random.default_rng(seed=SEED)


@pytest.fixture
def urng(xp: ModuleType) -> UnifiedGenerator:
    """
    Fixture for a unified RNG interface.

    Access the relevant RNG using `urng.` in tests.

    Must be used with the `xp` fixture. Use `rng` for non array API tests.
    """
    return _select_urng(xp)


@pytest.fixture
def urngb(xpb: ModuleType) -> UnifiedGenerator:
    """
    Fixture for a unified RNG interface to be used in benchmarks.

    Access the relevant RNG using `urngb.` in tests.

    Must be used with the `xpb` fixture.
    """
    return _select_urng(xpb)
