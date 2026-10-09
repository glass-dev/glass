from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import pytest

pytest.importorskip("jax", reason="tests require jax")

import jax.numpy as jnp
import jax.test_util

import array_api_compat

import glass

if TYPE_CHECKING:
    from glass._types import FloatArray, UnifiedGenerator


@pytest.mark.parametrize("count", [[10], [20, 2]])
def test_draw_nz(count: list[int]) -> None:
    """Tests glass.galaxies._draw_nz is auto differentiable when using JAX."""
    pytest.skip("glass.galaxies._draw_nz is not auto differentiable")

    z = jnp.asarray([0.0, 1.0, 2.0, 3.0, 4.0])
    nz = jnp.asarray([5.0, 6.0, 7.0, 8.0, 9.0])
    rng = glass.rng.default_rng(xp=jnp)

    def _draw_nz_by_z_and_nz(z: FloatArray, nz: FloatArray) -> FloatArray:
        # Need to ensure all inputs use the same arrays types here otherwise we end up
        # with a mix of backends as `check_grads`` perturbs `z`` and `nz`` using NumPy.
        xp = array_api_compat.array_namespace(z, nz, use_compat=False)
        return glass.galaxies._draw_nz(
            xp.asarray(count),
            z,
            nz,
            rng=rng,
        )

    jax.test_util.check_grads(
        _draw_nz_by_z_and_nz,
        (z, nz),
        order=1,
    )


def test_redshifts_from_bins() -> None:
    """Tests glass.redshifts_from_bins is auto differentiable when using JAX."""
    z = jnp.asarray([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])

    def redshifts_from_bins_by_z(z: FloatArray) -> FloatArray:
        # Must supply a fresh JAX RNG so that we generate the same sequence for each
        # call carried out by jax.test_util.check_grads
        rng = glass.rng.default_rng(xp=jnp)
        # Need to ensure all inputs use the same arrays types here otherwise we end up
        # with a mix of backends as `check_grads` perturbs `z` using NumPy.
        xp = z.__array_namespace__()
        bins = xp.asarray([5, 1, 4, 2])
        nz_dict = {
            1: xp.asarray([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            2: xp.asarray([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            4: xp.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
            5: xp.asarray([0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
        }
        return glass.redshifts_from_bins(bins, z, nz_dict, rng=rng)

    jax.test_util.check_grads(
        redshifts_from_bins_by_z,
        (z,),
        order=1,
    )


@pytest.mark.parametrize("count", [[10], [10, 20, 30]])
def test_redshifts_from_nz(count: list[int]) -> None:
    """Tests glass.redshifts_from_nz is auto differentiable when using JAX."""
    z = jnp.asarray([0, 1, 2, 3, 4], dtype=jnp.float64)
    nz = jnp.asarray([1, 0, 0, 0, 0], dtype=jnp.float64)

    def redshifts_from_nz_by_z_and_nz(z: FloatArray, nz: FloatArray) -> FloatArray:
        # Must supply a fresh JAX RNG so that we generate the same sequence for each
        # call carried out by jax.test_util.check_grads
        rng = glass.rng.default_rng(xp=jnp)
        # Need to ensure all inputs use the same arrays types here otherwise we end up
        # with a mix of backends as `check_grads` perturbs `z` and `nz` using NumPy.
        xp = z.__array_namespace__()
        return glass.redshifts_from_nz(xp.asarray(count), z, nz, rng=rng, warn=False)

    jax.test_util.check_grads(
        redshifts_from_nz_by_z_and_nz,
        (z, nz),
        order=1,
    )


@pytest.mark.parametrize("reduced_shear", [True, False])
def test_galaxy_shear(rng: UnifiedGenerator, reduced_shear: bool) -> None:
    """Tests glass.galaxy_shear is auto differentiable when using JAX."""
    pytest.skip("glass.galaxy_shear is not auto differentiable")

    kappa = rng.normal(size=(12,))
    gamma1 = rng.normal(size=(12,))
    gamma2 = rng.normal(size=(12,))
    gal_lon = rng.normal(size=(512,))
    gal_lat = rng.normal(size=(512,))
    gal_eps = rng.normal(size=(512,))

    jax.test_util.check_grads(
        partial(glass.galaxy_shear, reduced_shear=reduced_shear),
        (gal_lon, gal_lat, gal_eps, kappa, gamma1, gamma2),
        order=1,
    )


@pytest.mark.parametrize(
    ("z", "sigma_0"),
    [
        (1.0, jnp.ones((11, 1))),
        (jnp.linspace(0, 1, 5), 1.0),
        (jnp.linspace(0, 1, 5), jnp.ones((11, 1))),
    ],
)
@pytest.mark.parametrize("lower", [0.0, None])
@pytest.mark.parametrize("upper", [1.0, None])
def test_gaussian_phz_no_bounds(
    lower: float | None,
    sigma_0: float | FloatArray,
    upper: float | None,
    z: float | FloatArray,
) -> None:
    """Tests glass.gaussian_phz is auto differentiable when using JAX."""

    def gaussian_phz_arrays_only(
        z: float | FloatArray, sigma_0: int | FloatArray
    ) -> FloatArray:
        # Must supply a fresh JAX RNG so that we generate the same sequence for each
        # call carried out by jax.test_util.check_grads
        rng = glass.rng.default_rng(xp=jnp)
        return glass.gaussian_phz(z, sigma_0, lower=lower, upper=upper, rng=rng)

    jax.test_util.check_grads(
        gaussian_phz_arrays_only,
        (z, sigma_0),
        order=1,
    )
