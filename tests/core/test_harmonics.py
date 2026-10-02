from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

import pytest

import array_api_extra as xpx

import glass.harmonics

if TYPE_CHECKING:
    from types import ModuleType

    from glass._types import UnifiedGenerator
    from tests.fixtures.helper_classes import HealpixInputs

# check if available for testing
HAVE_ARRAY_API_STRICT = importlib.util.find_spec("array_api_strict") is not None
HAVE_JAX = importlib.util.find_spec("jax") is not None
HAVE_S2FFT = importlib.util.find_spec("s2fft") is not None


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


@pytest.mark.skipif(not (HAVE_JAX and HAVE_S2FFT), reason="test requires jax and s2fft")
def test_inverse_transform(
    healpix_inputs: type[HealpixInputs],
    rng: UnifiedGenerator,
) -> None:
    import jax
    import jax.numpy as jnp

    alm = healpix_inputs.alm(rng=rng)

    with jax.enable_x64(True):  # noqa: FBT003
        expected = glass.harmonics.inverse_transform(
            alm,
            lmax=healpix_inputs.lmax,
            nside=healpix_inputs.nside,
        )
        actual = glass.harmonics.inverse_transform(
            jnp.asarray(alm),
            lmax=healpix_inputs.lmax,
            nside=healpix_inputs.nside,
        )
        xpx.testing.assert_close(actual, jnp.asarray(expected), atol=1e-12, rtol=0)


@pytest.mark.skipif(not (HAVE_JAX and HAVE_S2FFT), reason="test requires jax and s2fft")
def test_transform(
    healpix_inputs: type[HealpixInputs],
    rng: UnifiedGenerator,
) -> None:
    import jax
    import jax.numpy as jnp

    kappa = healpix_inputs.kappa(rng=rng)

    with jax.enable_x64(True):  # noqa: FBT003
        expected = glass.harmonics.transform(
            kappa,
            lmax=healpix_inputs.lmax,
            nside=healpix_inputs.nside,
        )
        actual = glass.harmonics.transform(
            jnp.asarray(kappa),
            lmax=healpix_inputs.lmax,
            nside=healpix_inputs.nside,
        )
        xpx.testing.assert_close(actual, jnp.asarray(expected), atol=1e-15, rtol=0)
