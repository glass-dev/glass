from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

jax = pytest.importorskip("jax", reason="tests require jax")
import jax.test_util

import glass
from glass import rng

if TYPE_CHECKING:
    from glass._types import UnifiedGenerator

jnp = jax.numpy

@pytest.fixture(scope="session")
def urng() -> UnifiedGenerator:
    return rng.default_rng(xp=jnp)

def test_compute():
    """Tests that glass.grf.compute is auto differentiable when using JAX."""
    t1 = glass.grf.Normal()
    t2 = glass.grf.Normal()
    x = jnp.zeros(10)

    def compute_by_x(x):
        return glass.grf.compute(x, t1, t2)

    jax.test_util.check_grads(
        compute_by_x,
        (x,),
        order=1,
    )
