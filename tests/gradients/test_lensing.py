from __future__ import annotations

from typing import TYPE_CHECKING

import jax.test_util
import pytest

import glass
import glass.cosmology
import glass.healpix as hp

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

    from glass._types import ComplexArray, FloatArray, UnifiedGenerator
    from glass.cosmology import Cosmology


@pytest.mark.parametrize("potential", [True, False])
@pytest.mark.parametrize("deflection", [True, False])
@pytest.mark.parametrize("shear", [True, False])
@pytest.mark.parametrize("discretized", [True, False])
def test_from_convergence(  # noqa: PLR0913, PLR0917
    jnp: ModuleType,
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

    def from_convergence_diffable(
        kappa: FloatArray,
    ) -> tuple[FloatArray | ComplexArray, ...]:
        return glass.from_convergence(
            jnp.asarray(kappa),
            potential=potential,
            deflection=deflection,
            shear=shear,
            discretized=discretized,
        )

    jax.test_util.check_grads(
        from_convergence_diffable,
        (kappa,),
        order=1,
    )


def test_multi_plane_weights(
    get_cosmo: Callable[..., Cosmology],
    get_shells: Callable[..., list[glass.RadialWindow]],
    jnp: ModuleType,
) -> None:
    """Tests glass.multi_plane_weights is auto differentiable when using JAX."""
    shells = get_shells(jnp)
    weights = jnp.eye(len(shells))
    cosmo = get_cosmo(jnp)

    def multi_plane_weights_diffable(weights: FloatArray) -> FloatArray:
        return glass.multi_plane_weights(
            jnp.asarray(weights),
            shells=shells,
            cosmo=cosmo,
        )

    jax.test_util.check_grads(
        multi_plane_weights_diffable,
        (weights,),
        order=1,
    )
