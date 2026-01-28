from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest

import glass
import glass.healpix as hp
from glass._array_api_utils import xp_additions as uxpx

if TYPE_CHECKING:
    from types import ModuleType

    from glass._types import FloatArray, UnifiedGenerator
    from glass.cosmology import Cosmology
    from tests.fixtures.helper_classes import Compare


def test_from_convergence(
    compare: type[Compare],
    urng: UnifiedGenerator,
) -> None:
    """Add unit tests for :func:`glass.from_convergence`."""
    # l_max = 100  # noqa: ERA001
    n_side = 32

    # create a convergence map
    kappa = urng.random(hp.nside2npix(n_side))
    kappa *= 10.0

    # check with all False

    results = glass.from_convergence(kappa)
    compare.assert_array_equal(results, ())

    # check all combinations of potential, deflection, shear being True

    results = glass.from_convergence(kappa, potential=True)
    compare.assert_array_equal(len(results), 1)

    results = glass.from_convergence(kappa, deflection=True)
    compare.assert_array_equal(len(results), 1)

    results = glass.from_convergence(kappa, shear=True)
    compare.assert_array_equal(len(results), 1)

    results = glass.from_convergence(kappa, potential=True, deflection=True)
    compare.assert_array_equal(len(results), 2)

    results = glass.from_convergence(kappa, potential=True, shear=True)
    compare.assert_array_equal(len(results), 2)

    results = glass.from_convergence(kappa, deflection=True, shear=True)
    compare.assert_array_equal(len(results), 2)

    results = glass.from_convergence(kappa, potential=True, deflection=True, shear=True)
    compare.assert_array_equal(len(results), 3)


def test_shear_from_convergence() -> None:
    """Add unit tests for :func:`glass.shear_from_convergence`."""
    pytest.skip("No test yet implemented")


def test_multi_plane_matrix(
    compare: type[Compare],
    cosmo: Cosmology,
    shells: list[glass.RadialWindow],
    urng: UnifiedGenerator,
    xp: ModuleType,
) -> None:
    mat = glass.multi_plane_matrix(shells, cosmo)

    compare.assert_array_equal(mat, xp.tril(mat))
    compare.assert_array_equal(xp.triu(mat, k=1), 0)

    convergence = glass.MultiPlaneConvergence(cosmo)

    deltas = urng.random((len(shells), 10))
    kappas = []
    for i in range(len(shells)):
        shell = shells[i]
        delta = deltas[i, ...]
        convergence.add_window(delta, shell)
        if convergence.kappa is not None:
            kappas.append(xp.asarray(convergence.kappa, copy=True))

    compare.assert_allclose(mat @ deltas, kappas)


def test_multi_plane_weights(
    compare: type[Compare],
    cosmo: Cosmology,
    urng: UnifiedGenerator,
    shells: list[glass.RadialWindow],
    xp: ModuleType,
) -> None:
    w_in = xp.eye(len(shells))
    w_out = glass.multi_plane_weights(w_in, shells, cosmo)

    compare.assert_array_equal(w_out, xp.triu(w_out, k=1))
    compare.assert_array_equal(xp.tril(w_out), 0)

    convergence = glass.MultiPlaneConvergence(cosmo)

    deltas = urng.random((len(shells), 10))
    weights = urng.random((len(shells), 3))
    kappa = 0
    for i in range(min(len(shells), deltas.shape[0], weights.shape[0])):
        shell = shells[i]
        delta = deltas[i, :]
        weight = weights[i, :]
        convergence.add_window(delta, shell)
        assert convergence.kappa is not None
        kappa = kappa + weight[..., xp.newaxis] * convergence.kappa
    kappa /= xp.sum(weights, axis=0)[..., xp.newaxis]

    wmat = glass.multi_plane_weights(weights, shells, cosmo)

    compare.assert_allclose(uxpx.einsum("ij,ik", wmat, deltas), kappa)


@pytest.mark.parametrize("usecomplex", [True, False])
def test_deflect_nsew(
    compare: type[Compare],
    usecomplex: bool,  # noqa: FBT001
    xp: ModuleType,
) -> None:
    d = 5.0
    r = math.radians(d)

    def alpha(
        re: float,
        im: float,
        *,
        usecomplex: bool,
    ) -> complex | FloatArray:
        return re + 1j * im if usecomplex else xp.asarray([re, im])

    # north
    lon, lat = glass.deflect(0.0, 0.0, alpha(r, 0, usecomplex=usecomplex), xp=xp)
    compare.assert_allclose([lon, lat], [0.0, d], atol=1e-15)

    # south
    lon, lat = glass.deflect(0.0, 0.0, alpha(-r, 0, usecomplex=usecomplex), xp=xp)
    compare.assert_allclose([lon, lat], [0.0, -d], atol=1e-15)

    # east
    lon, lat = glass.deflect(0.0, 0.0, alpha(0, r, usecomplex=usecomplex), xp=xp)
    compare.assert_allclose([lon, lat], [-d, 0.0], atol=1e-15)

    # west
    lon, lat = glass.deflect(0.0, 0.0, alpha(0, -r, usecomplex=usecomplex), xp=xp)
    compare.assert_allclose([lon, lat], [d, 0.0], atol=1e-15)

    # At least one input is an array
    lon, lat = glass.deflect(
        xp.asarray(0.0),
        xp.asarray(0.0),
        alpha(0, -r, usecomplex=usecomplex),
    )
    compare.assert_allclose([lon, lat], [d, 0.0], atol=1e-15)

    lon, lat = glass.deflect(
        xp.asarray([0.0, 0.0]),
        xp.asarray([0.0, 0.0]),
        alpha(0, -r, usecomplex=usecomplex),
    )
    compare.assert_allclose(lon, xp.asarray([d, d]), atol=1e-15)
    compare.assert_allclose(lat, 0.0, atol=1e-15)

    # No inputs are arrays and xp not provided
    with pytest.raises(
        TypeError,
        match="array_namespace requires at least one non-scalar array input",
    ):
        glass.deflect(0.0, 0.0, alpha(0, -r, usecomplex=True))


def test_deflect_many(
    compare: type[Compare],
    urng: UnifiedGenerator,
    xp: ModuleType,
) -> None:
    n = 1_000
    abs_alpha = urng.uniform(0, 2 * math.pi, size=n)
    arg_alpha = urng.uniform(-math.pi, math.pi, size=n)

    lon_ = uxpx.degrees(urng.uniform(-math.pi, math.pi, size=n))
    lat_ = uxpx.degrees(xp.asin(urng.uniform(-1, 1, size=n)))

    lon, lat = glass.deflect(lon_, lat_, abs_alpha * xp.exp(1j * arg_alpha))

    x_, y_, z_ = hp.ang2vec(lon_, lat_, lonlat=True, xp=xp)
    x, y, z = hp.ang2vec(lon, lat, lonlat=True, xp=xp)

    dotp = x * x_ + y * y_ + z * z_

    compare.assert_allclose(dotp, xp.cos(abs_alpha))
