from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import pytest

pytest.importorskip("jax", reason="tests require jax")

import jax.numpy as jnp
import jax.test_util

import glass
import glass.cosmology
import glass.healpix as hp

if TYPE_CHECKING:
    from collections.abc import Callable

    from glass._types import FloatArray, UnifiedGenerator
    from glass.cosmology import Cosmology


@pytest.mark.parametrize("potential", [True, False])
@pytest.mark.parametrize("deflection", [True, False])
@pytest.mark.parametrize("shear", [True, False])
@pytest.mark.parametrize("discretized", [True, False])
def test_from_convergence(
    rng: UnifiedGenerator,
    potential: bool,
    deflection: bool,
    shear: bool,
    discretized: bool,
) -> None:
    """Tests glass.from_convergence is auto differentiable when using JAX."""
    pytest.skip("glass.from_convergence is not auto differentiable")

    # create a convergence map
    kappa = rng.random(hp.nside2npix(nside=32))
    kappa *= 10.0

    jax.test_util.check_grads(
        partial(
            glass.from_convergence,
            potential=potential,
            deflection=deflection,
            shear=shear,
            discretized=discretized,
        ),
        (kappa,),
        order=1,
    )


def test_multi_plane_weights(
    cosmo: Cosmology,
    get_shells: Callable[..., list[glass.RadialWindow]],
) -> None:
    """Tests glass.multi_plane_weights is auto differentiable when using JAX."""
    shells = get_shells(jnp)
    weights = jnp.eye(len(shells))

    def multi_plane_weights_by_weights(weights: FloatArray) -> FloatArray:
        # Need to ensure all inputs use the same arrays types here otherwise we end up
        # with a mix of backends as `check_grads`` perturbs weights using NumPy.
        xp = weights.__array_namespace__()
        shells = get_shells(xp)
        return glass.multi_plane_weights(
            weights,
            shells=shells,
            cosmo=cosmo,
        )

    jax.test_util.check_grads(
        multi_plane_weights_by_weights,
        (weights,),
        order=1,
    )
