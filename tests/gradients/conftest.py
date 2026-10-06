from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import pytest

import glass.rng

if TYPE_CHECKING:
    from glass._types import UnifiedGenerator


@pytest.fixture
def rng() -> UnifiedGenerator:
    """RNG fixture in gradient tests."""
    return glass.rng.default_rng(xp=jnp)
