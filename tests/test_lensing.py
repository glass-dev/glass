

from typing import TYPE_CHECKING

import healpix
import numpy as np
import pytest

import glass

if TYPE_CHECKING:
    from glass.cosmology import Cosmology


def test_from_convergence(rng: np.random.Generator) -> None:
    """Add unit tests for :func:`glass.from_convergence`."""
    # l_max = 100  # noqa: ERA001
    n_side = 32

    # create a convergence map
    kappa = rng.integers(10, size=healpix.nside2npix(n_side))

    # check with all False

    results = glass.from_convergence(kappa)
    np.testing.assert_array_equal(results, ())

    # check all combinations of potential, deflection, shear being True

    results = glass.from_convergence(kappa, potential=True)
    np.testing.assert_array_equal(len(results), 1)

    results = glass.from_convergence(kappa, deflection=True)
    np.testing.assert_array_equal(len(results), 1)

    results = glass.from_convergence(kappa, shear=True)
    np.testing.assert_array_equal(len(results), 1)

    results = glass.from_convergence(kappa, potential=True, deflection=True)
    np.testing.assert_array_equal(len(results), 2)

    results = glass.from_convergence(kappa, potential=True, shear=True)
    np.testing.assert_array_equal(len(results), 2)

    results = glass.from_convergence(kappa, deflection=True, shear=True)
    np.testing.assert_array_equal(len(results), 2)

    results = glass.from_convergence(kappa, potential=True, deflection=True, shear=True)
    np.testing.assert_array_equal(len(results), 3)


def test_shear_from_convergence() -> None:
    """Add unit tests for :func:`glass.shear_from_convergence`."""


def test_multi_plane_matrix(
    shells: list[glass.RadialWindow],
    cosmo: Cosmology,
    rng: np.random.Generator,
) -> None:
    mat = glass.multi_plane_matrix(shells, cosmo)

    np.testing.assert_array_equal(mat, np.tril(mat))
    np.testing.assert_array_equal(np.triu(mat, 1), 0)

    convergence = glass.MultiPlaneConvergence(cosmo)

    deltas = rng.random((len(shells), 10))
    kappas = []
    for shell, delta in zip(shells, deltas, strict=False):
        convergence.add_window(delta, shell)
        if convergence.kappa is not None:
            kappas.append(convergence.kappa.copy())

    np.testing.assert_allclose(mat @ deltas, kappas)


def test_multi_plane_weights(
    shells: list[glass.RadialWindow],
    cosmo: Cosmology,
    rng: np.random.Generator,
) -> None:
    w_in = np.eye(len(shells))
    w_out = glass.multi_plane_weights(w_in, shells, cosmo)

    np.testing.assert_array_equal(w_out, np.triu(w_out, 1))
    np.testing.assert_array_equal(np.tril(w_out), 0)

    convergence = glass.MultiPlaneConvergence(cosmo)

    deltas = rng.random((len(shells), 10))
    weights = rng.random((len(shells), 3))
    kappa = 0
    for shell, delta, weight in zip(shells, deltas, weights, strict=False):
        convergence.add_window(delta, shell)
        kappa = kappa + weight[..., None] * convergence.kappa
    kappa /= weights.sum(axis=0)[..., None]

    wmat = glass.multi_plane_weights(weights, shells, cosmo)

    np.testing.assert_allclose(np.einsum("ij,ik", wmat, deltas), kappa)


@pytest.mark.parametrize("usecomplex", [True, False])
def test_deflect_nsew(usecomplex: bool) -> None:  # noqa: FBT001
    d = 5.0
    r = np.radians(d)

    def alpha(re: float, im: float, *, usecomplex: bool) -> complex | list[float]:
        return re + 1j * im if usecomplex else [re, im]

    # north
    lon, lat = glass.deflect(0.0, 0.0, alpha(r, 0, usecomplex=usecomplex))
    np.testing.assert_allclose([lon, lat], [0.0, d], atol=1e-15)

    # south
    lon, lat = glass.deflect(0.0, 0.0, alpha(-r, 0, usecomplex=usecomplex))
    np.testing.assert_allclose([lon, lat], [0.0, -d], atol=1e-15)

    # east
    lon, lat = glass.deflect(0.0, 0.0, alpha(0, r, usecomplex=usecomplex))
    np.testing.assert_allclose([lon, lat], [-d, 0.0], atol=1e-15)

    # west
    lon, lat = glass.deflect(0.0, 0.0, alpha(0, -r, usecomplex=usecomplex))
    np.testing.assert_allclose([lon, lat], [d, 0.0], atol=1e-15)


def test_deflect_many(rng: np.random.Generator) -> None:
    n = 1000
    abs_alpha = rng.uniform(0, 2 * np.pi, size=n)
    arg_alpha = rng.uniform(-np.pi, np.pi, size=n)

    lon_ = np.degrees(rng.uniform(-np.pi, np.pi, size=n))
    lat_ = np.degrees(np.arcsin(rng.uniform(-1, 1, size=n)))

    lon, lat = glass.deflect(lon_, lat_, abs_alpha * np.exp(1j * arg_alpha))

    x_, y_, z_ = healpix.ang2vec(lon_, lat_, lonlat=True)
    x, y, z = healpix.ang2vec(lon, lat, lonlat=True)

    dotp = x * x_ + y * y_ + z * z_

    np.testing.assert_allclose(dotp, np.cos(abs_alpha))
