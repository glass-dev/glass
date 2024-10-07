import healpix
import numpy as np
import pytest

from glass.lensing import (
    MultiPlaneConvergence,
    deflect,
    multi_plane_matrix,
    multi_plane_weights,
)
from glass.shells import RadialWindow


@pytest.fixture
def shells() -> list[RadialWindow]:
    return [
        RadialWindow([0.0, 1.0, 2.0], [0.0, 1.0, 0.0], 1.0),
        RadialWindow([1.0, 2.0, 3.0], [0.0, 1.0, 0.0], 2.0),
        RadialWindow([2.0, 3.0, 4.0], [0.0, 1.0, 0.0], 3.0),
        RadialWindow([3.0, 4.0, 5.0], [0.0, 1.0, 0.0], 4.0),
        RadialWindow([4.0, 5.0, 6.0], [0.0, 1.0, 0.0], 5.0),
    ]


@pytest.fixture
def cosmo():
    class MockCosmology:
        @property
        def omega_m(self):
            return 0.3

        def ef(self, z):
            return (self.omega_m * (1 + z) ** 3 + 1 - self.omega_m) ** 0.5

        def xm(self, z, z2=None):
            if z2 is None:
                return np.array(z) * 1000
            return (np.array(z2) - np.array(z)) * 1000

    return MockCosmology()


@pytest.mark.parametrize("usecomplex", [True, False])
def test_deflect_nsew(usecomplex: bool) -> None:
    d = 5.0
    r = np.radians(d)

    if usecomplex:

        def alpha(re, im):
            return re + 1j * im
    else:

        def alpha(re, im):
            return [re, im]

    # north
    lon, lat = deflect(0.0, 0.0, alpha(r, 0))
    assert np.allclose([lon, lat], [0.0, d])

    # south
    lon, lat = deflect(0.0, 0.0, alpha(-r, 0))
    assert np.allclose([lon, lat], [0.0, -d])

    # east
    lon, lat = deflect(0.0, 0.0, alpha(0, r))
    assert np.allclose([lon, lat], [-d, 0.0])

    # west
    lon, lat = deflect(0.0, 0.0, alpha(0, -r))
    assert np.allclose([lon, lat], [d, 0.0])


def test_deflect_many(rng) -> None:
    n = 1000
    abs_alpha = rng.uniform(0, 2 * np.pi, size=n)
    arg_alpha = rng.uniform(-np.pi, np.pi, size=n)

    lon_ = np.degrees(rng.uniform(-np.pi, np.pi, size=n))
    lat_ = np.degrees(np.arcsin(rng.uniform(-1, 1, size=n)))

    lon, lat = deflect(lon_, lat_, abs_alpha * np.exp(1j * arg_alpha))

    x_, y_, z_ = healpix.ang2vec(lon_, lat_, lonlat=True)
    x, y, z = healpix.ang2vec(lon, lat, lonlat=True)

    dotp = x * x_ + y * y_ + z * z_

    np.testing.assert_allclose(dotp, np.cos(abs_alpha))


def test_multi_plane_matrix(shells, cosmo, rng) -> None:
    mat = multi_plane_matrix(shells, cosmo)

    np.testing.assert_array_equal(mat, np.tril(mat))
    np.testing.assert_array_equal(np.triu(mat, 1), 0)

    convergence = MultiPlaneConvergence(cosmo)

    deltas = rng.random((len(shells), 10))
    kappas = []
    for shell, delta in zip(shells, deltas):
        convergence.add_window(delta, shell)
        kappas.append(convergence.kappa.copy())

    np.testing.assert_allclose(mat @ deltas, kappas)


def test_multi_plane_weights(shells, cosmo, rng) -> None:
    w_in = np.eye(len(shells))
    w_out = multi_plane_weights(w_in, shells, cosmo)

    np.testing.assert_array_equal(w_out, np.triu(w_out, 1))
    np.testing.assert_array_equal(np.tril(w_out), 0)

    convergence = MultiPlaneConvergence(cosmo)

    deltas = rng.random((len(shells), 10))
    weights = rng.random((len(shells), 3))
    kappa = 0
    for shell, delta, weight in zip(shells, deltas, weights):
        convergence.add_window(delta, shell)
        kappa = kappa + weight[..., None] * convergence.kappa
    kappa /= weights.sum(axis=0)[..., None]

    wmat = multi_plane_weights(weights, shells, cosmo)

    np.testing.assert_allclose(np.einsum("ij,ik", wmat, deltas), kappa)
