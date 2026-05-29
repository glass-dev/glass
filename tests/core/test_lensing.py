from __future__ import annotations

import math
import typing

import pytest

import array_api_extra as xpx

import glass
import glass.healpix as hp
from glass._array_api_utils import xp_additions as uxpx

if typing.TYPE_CHECKING:
    from types import ModuleType

    from glass._types import FloatArray, UnifiedGenerator
    from glass.cosmology import Cosmology


def test_from_convergence(urng: UnifiedGenerator) -> None:
    """Add unit tests for :func:`glass.from_convergence`."""
    # l_max = 100  # noqa: ERA001
    n_side = 32

    # create a convergence map
    kappa = urng.random(hp.nside2npix(n_side))
    kappa *= 10.0

    # check with all False

    results = glass.from_convergence(kappa)  # ty: ignore[no-matching-overload]
    assert results == ()

    # check all combinations of potential, deflection, shear being True

    results = glass.from_convergence(kappa, potential=True)
    assert len(results) == 1

    results = glass.from_convergence(kappa, deflection=True)
    assert len(results) == 1

    results = glass.from_convergence(kappa, shear=True)
    assert len(results) == 1

    results = glass.from_convergence(kappa, potential=True, deflection=True)
    assert len(results) == 2

    results = glass.from_convergence(kappa, potential=True, shear=True)
    assert len(results) == 2

    results = glass.from_convergence(kappa, deflection=True, shear=True)
    assert len(results) == 2

    results = glass.from_convergence(kappa, potential=True, deflection=True, shear=True)
    assert len(results) == 3


def test_shear_from_convergence() -> None:
    """Add unit tests for :func:`glass.shear_from_convergence`."""
    pytest.skip("No test yet implemented")


def test_multi_plane_matrix(
    cosmo: Cosmology,
    shells: list[glass.RadialWindow],
    urng: UnifiedGenerator,
    xp: ModuleType,
) -> None:
    mat = glass.multi_plane_matrix(shells, cosmo)

    xpx.testing.assert_equal(mat, xp.tril(mat))
    xpx.testing.assert_equal(xp.triu(mat, k=1), xp.asarray(0.0), check_shape=False)

    convergence = glass.MultiPlaneConvergence(cosmo)

    deltas = urng.random((len(shells), 10))
    kappas = []
    for i in range(len(shells)):
        shell = shells[i]
        delta = deltas[i, ...]
        convergence.add_window(delta, shell)
        if convergence.kappa is not None:
            kappas.append(xp.asarray(convergence.kappa, copy=True))

    xpx.testing.assert_close(mat @ deltas, xp.stack(kappas))


def test_multi_plane_weights(
    cosmo: Cosmology,
    urng: UnifiedGenerator,
    shells: list[glass.RadialWindow],
    xp: ModuleType,
) -> None:
    w_in = xp.eye(len(shells))
    w_out = glass.multi_plane_weights(w_in, shells, cosmo)

    xpx.testing.assert_equal(w_out, xp.triu(w_out, k=1))
    xpx.testing.assert_equal(xp.tril(w_out), xp.asarray(0.0), check_shape=False)

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

    xpx.testing.assert_close(uxpx.einsum("ij,ik", wmat, deltas), kappa)


@pytest.mark.parametrize("usecomplex", [True, False])
def test_deflect_nsew(
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
    xpx.testing.assert_close(xp.stack([lon, lat]), xp.asarray([0.0, d]), atol=1e-15)

    # south
    lon, lat = glass.deflect(0.0, 0.0, alpha(-r, 0, usecomplex=usecomplex), xp=xp)
    xpx.testing.assert_close(xp.stack([lon, lat]), xp.asarray([0.0, -d]), atol=1e-15)

    # east
    lon, lat = glass.deflect(0.0, 0.0, alpha(0, r, usecomplex=usecomplex), xp=xp)
    xpx.testing.assert_close(xp.stack([lon, lat]), xp.asarray([-d, 0.0]), atol=1e-15)

    # west
    lon, lat = glass.deflect(0.0, 0.0, alpha(0, -r, usecomplex=usecomplex), xp=xp)
    xpx.testing.assert_close(xp.stack([lon, lat]), xp.asarray([d, 0.0]), atol=1e-15)

    # At least one input is an array
    lon, lat = glass.deflect(
        xp.asarray(0.0),
        xp.asarray(0.0),
        alpha(0, -r, usecomplex=usecomplex),
    )
    xpx.testing.assert_close(xp.stack([lon, lat]), xp.asarray([d, 0.0]), atol=1e-15)

    lon, lat = glass.deflect(
        xp.asarray([0.0, 0.0]),
        xp.asarray([0.0, 0.0]),
        alpha(0, -r, usecomplex=usecomplex),
    )
    xpx.testing.assert_close(lon, xp.asarray([d, d]), atol=1e-15)
    xpx.testing.assert_close(lat, xp.asarray(0.0), atol=1e-15, check_shape=False)

    # No inputs are arrays and xp not provided
    with pytest.raises(
        TypeError,
        match="array_namespace requires at least one non-scalar array input",
    ):
        glass.deflect(0.0, 0.0, alpha(0, -r, usecomplex=True))


def test_deflect_many(
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

    xpx.testing.assert_close(dotp, xp.cos(abs_alpha))
