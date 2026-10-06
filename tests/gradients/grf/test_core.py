from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

pytest.importorskip("jax", reason="tests require jax")

import jax.test_util

import glass

if TYPE_CHECKING:
    from types import ModuleType

    from glass._types import AnyArray


def test_compute(jnp: ModuleType) -> None:
    """Tests that glass.grf.compute is auto differentiable when using JAX."""
    t1 = glass.grf.Normal()
    t2 = glass.grf.Normal()
    x = jnp.zeros(10)

    def compute_by_x(x: AnyArray) -> AnyArray:
        return glass.grf.compute(x, t1, t2)

    jax.test_util.check_grads(
        compute_by_x,
        (x,),
        order=1,
    )
