from __future__ import annotations

from typing import TYPE_CHECKING

import jax.test_util
import pytest

import glass

if TYPE_CHECKING:
    from types import ModuleType

    from glass._types import FloatArray, UnifiedGenerator


def test_triaxial_axis_ratio(
    jnp: ModuleType,
    rng: UnifiedGenerator,
) -> None:
    """Tests glass.triaxial_axis_ratio is auto differentiable."""
    pytest.skip("glass.triaxial_axis_ratio is not auto differentiable")

    sorted_uniform_rnd = jnp.sort(
        rng.uniform(
            0,
            1,
            size=(2, 1_000),
        ),
        axis=0,
    )
    zeta = sorted_uniform_rnd[0, :]
    xi = sorted_uniform_rnd[1, :]

    def triaxial_axis_ratio_same_rng(
        zeta: FloatArray,
        xi: FloatArray,
    ) -> FloatArray:
        fresh_rng = glass.rng.default_rng(xp=jnp)
        return glass.triaxial_axis_ratio(zeta, xi, rng=fresh_rng, xp=jnp)

    jax.test_util.check_grads(
        triaxial_axis_ratio_same_rng,
        (zeta, xi),
        order=1,
    )
