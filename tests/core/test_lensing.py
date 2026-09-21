from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import array_api_extra as xpx

import glass
import glass.healpix as hp
from glass._array_api_utils import xp_additions as uxpx

if TYPE_CHECKING:
    from types import ModuleType

    from glass._types import UnifiedGenerator
    from glass.cosmology import Cosmology


def test_from_convergence(urng: UnifiedGenerator) -> None:
    """Add unit tests for :func:`glass.from_convergence`."""
    # l_max = 100  # noqa: ERA001
    n_side = 32

    # create a convergence map
    kappa = urng.random(hp.nside2npix(n_side))
    kappa *= 10.0

    # check with all False

    results = glass.from_convergence(kappa)
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
