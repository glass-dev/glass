from __future__ import annotations

import math
from typing import TYPE_CHECKING

import healpy as hp
import pytest
import healpix

import array_api_extra as xpx

import glass.harmonics

if TYPE_CHECKING:
    from types import ModuleType

    from tests.fixtures.helper_classes import Compare


def test_multalm(compare: type[Compare], xp: ModuleType) -> None:
    # Call jax version of iternorm once jax version is written
    if xp.__name__ == "jax.numpy":
        pytest.skip("Arrays in multalm are not immutable, so do not support jax")

    # check output values and shapes

    alm = xp.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    bl = xp.asarray([2.0, 0.5, 1.0])
    alm_copy = xp.asarray(alm, copy=True)

    result = glass.harmonics.multalm(alm, bl)

    expected_result = xp.asarray([2.0, 1.0, 1.5, 4.0, 5.0, 6.0])
    compare.assert_allclose(result, expected_result)
    with pytest.raises(AssertionError, match="Not equal to tolerance"):
        compare.assert_allclose(alm_copy, result)

    # multiple with 1s

    alm = xp.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    bl = xp.ones(3)

    result = glass.harmonics.multalm(alm, bl)
    compare.assert_allclose(result, alm)

    # multiple with 0s

    bl = xp.asarray([0.0, 1.0, 0.0])

    result = glass.harmonics.multalm(alm, bl)

    expected_result = xp.asarray([0.0, 2.0, 3.0, 0.0, 0.0, 0.0])
    compare.assert_allclose(result, expected_result)

    # empty arrays

    alm = xp.asarray([])
    bl = xp.asarray([])

    result = glass.harmonics.multalm(alm, bl)
    compare.assert_allclose(result, alm)


def test_transform(compare: type[Compare], xp: ModuleType) -> None:
    if xp.__name__ in {"array_api_strict", "jax.numpy"}:
        pytest.skip("transform depends on healpy which is not Array API compatible")

    # prepare inputs
    constant = 1.23
    lmax = 0
    nside = 8

    # create an unpolarised map where every pixel is the same
    simple_map = xp.full(healpix.nside2npix(nside), constant)

    a00 = math.sqrt(4 * xp.pi) * constant
    # the alm array for lmax=0 has size 1, so it only contains a_00
    alm_expected = xp.asarray([a00], dtype=xp.complex128)

    alm_result = glass.harmonics.transform(
        simple_map,
        lmax=lmax,
        polarised_input=False,
    )

    compare.assert_allclose(alm_result, alm_expected, rtol=1e-15)


def test_inverse_transform(compare: type[Compare], xp: ModuleType) -> None:
    if xp.__name__ in {"array_api_strict", "jax.numpy"}:
        pytest.skip("transform depends on healpy which is not Array API compatible")

    # prepare inputs
    lmax = 1
    nside = 8
    spin = 1

    # create a simple monopole component ensuring a predictable output map value
    alm_size = hp.Alm.getsize(lmax)
    alm_monopole = xp.zeros(alm_size, dtype=xp.complex128)
    xpx.at(alm_monopole, 0).set(math.sqrt(4 * math.pi))

    # standard unpolarised transform without spin

    map_result = glass.harmonics.inverse_transform(
        alm_monopole,
        lmax=lmax,
        nside=nside,
        polarised_input=False,
    )

    assert hasattr(map_result, "ndim")
    assert map_result.ndim == 1
    compare.assert_array_equal(map_result, 1.0)

    # valid spin transform

    alm_input_list = [alm_monopole, alm_monopole]

    map_result_spin = glass.harmonics.inverse_transform(
        alm_input_list,
        lmax=lmax,
        nside=nside,
        spin=spin,
    )

    assert isinstance(map_result_spin, list)
    assert len(map_result_spin) == 2
    assert all(hasattr(m, "ndim") and m.ndim == 1 for m in map_result_spin)

    # invalid spin input

    with pytest.raises(
        ValueError, match="for spin transforms, alm must be of the form"
    ):
        glass.harmonics.inverse_transform(
            alm_monopole,
            lmax=lmax,
            nside=nside,
            spin=spin,
        )
