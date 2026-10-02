from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

import array_api_extra as xpx

import glass.harmonics

if TYPE_CHECKING:
    from types import ModuleType

    from glass._types import UnifiedGenerator
    from tests.fixtures.helper_classes import HealpixInputs


def test_multalm(xp: ModuleType) -> None:
    # check output values and shapes

    alm = xp.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    bl = xp.asarray([2.0, 0.5, 1.0])
    alm_copy = xp.asarray(alm, copy=True)

    result = glass.harmonics.multalm(alm, bl)

    expected_result = xp.asarray([2.0, 1.0, 3.0, 2.0, 5.0, 6.0])
    xpx.testing.assert_equal(result, expected_result)
    with pytest.raises(AssertionError, match="Not equal to tolerance"):
        xpx.testing.assert_close(alm_copy, result)

    # multiple with 1s

    alm = xp.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    bl = xp.ones(3)

    result = glass.harmonics.multalm(alm, bl)
    xpx.testing.assert_equal(result, alm)

    # multiple with 0s

    bl = xp.asarray([0.0, 1.0, 0.0])

    result = glass.harmonics.multalm(alm, bl)

    expected_result = xp.asarray([0.0, 2.0, 0.0, 4.0, 0.0, 0.0])
    xpx.testing.assert_equal(result, expected_result)

    # empty arrays

    alm = xp.asarray([])
    bl = xp.asarray([])

    result = glass.harmonics.multalm(alm, bl)
    xpx.testing.assert_equal(result, alm)


def test_inverse_transform(
    healpix_inputs: type[HealpixInputs],
    rng: UnifiedGenerator,
) -> None:
    jax = pytest.importorskip("jax", reason="test requires jax")

    import jax.numpy as jnp

    alm = healpix_inputs.alm(rng=rng)

    with jax.enable_x64(True):
        # https://github.com/data-apis/array-api-extra/issues/1005
        xpx.testing.assert_equal(
            glass.harmonics.inverse_transform(
                alm,
                lmax=healpix_inputs.lmax,
                nside=healpix_inputs.nside,
            ),
            np.asarray(
                glass.harmonics.inverse_transform(
                    jnp.asarray(alm),
                    lmax=healpix_inputs.lmax,
                    nside=healpix_inputs.nside,
                ),
            ),
        )


def test_transform(
    healpix_inputs: type[HealpixInputs],
    rng: UnifiedGenerator,
) -> None:
    jax = pytest.importorskip("jax", reason="test requires jax")

    import jax.numpy as jnp

    kappa = healpix_inputs.kappa(rng=rng)

    with jax.enable_x64(True):
        # https://github.com/data-apis/array-api-extra/issues/1005
        xpx.testing.assert_close(
            glass.harmonics.transform(
                kappa,
                lmax=healpix_inputs.lmax,
                nside=healpix_inputs.nside,
            ),
            np.asarray(
                glass.harmonics.transform(
                    jnp.asarray(kappa),
                    lmax=healpix_inputs.lmax,
                    nside=healpix_inputs.nside,
                ),
            ),
            atol=1e-15,
            rtol=0,
        )
