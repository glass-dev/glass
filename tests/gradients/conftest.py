from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import pytest

import glass.rng

if TYPE_CHECKING:
    from types import ModuleType

    from glass._types import UnifiedGenerator


jax.config.update("jax_enable_x64", val=True)


@pytest.fixture
def rng(jnp: ModuleType) -> UnifiedGenerator:
    """RNG fixture in gradient tests."""
    return glass.rng.default_rng(xp=jnp)
