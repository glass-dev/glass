from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

jax = pytest.importorskip("jax", reason="tests require jax")
import jax.test_util

import glass

if TYPE_CHECKING:
    from glass._types import AngularPowerSpectra, FloatArray, UnifiedGenerator

jnp = jax.numpy

@pytest.fixture(scope="session")
def cl() -> FloatArray:
    lmax = 100
    ell = jnp.arange(lmax + 1)
    return 1e-2 / (2 * ell + 1) ** 2

def test_one_transformation(cl: FloatArray) -> None:
    """Tests that glass.grf.solve is auto differentiable when using JAX."""
    pytest.skip("glass.grf.solve is not auto differentiable")
    lam = 0.12345
    t = glass.grf.Lognormal(lam)

    def solve_by_cl(cl):
        return glass.grf.solve(cl, t)

    jax.test_util.check_grads(
        solve_by_cl,
        (cl,),
        order=1,
    )


def test_pad(cl: FloatArray) -> None:
    """Tests that glass.grf.solve is auto differentiable when using JAX and passing pad."""
    pytest.skip("glass.grf.solve is not auto differentiable")
    lam = 0.12345
    t = glass.grf.Lognormal(lam)

    def solve_by_cl(cl):
        return glass.grf.solve(cl, t, pad=2 * cl.shape[0])

    jax.test_util.check_grads(
        solve_by_cl,
        (cl,),
        order=1,
    )


def test_initial(cl: FloatArray) -> None:
    """Tests that glass.grf.solve is auto differentiable when using JAX and passing initial."""
    pytest.skip("glass.grf.solve is not auto differentiable")
    lam = 0.12345
    t = glass.grf.Lognormal(lam)
    gl = glass.grf.compute(cl, t)

    def solve_by_cl(cl):
        return glass.grf.solve(cl, t, initial=gl)

    jax.test_util.check_grads(
        solve_by_cl,
        (cl,),
        order=1,
    )


def test_no_iterations(cl: FloatArray) -> None:
    """Tests that glass.grf.solve is auto differentiable when using JAX and no iterations."""
    pytest.skip("glass.grf.solve is not auto differentiable")
    lam = 0.12345
    t = glass.grf.Lognormal(lam)

    def solve_by_cl(cl):
        return glass.grf.solve(cl, t, maxiter=0)

    jax.test_util.check_grads(
        solve_by_cl,
        (cl,),
        order=1,
    )


def test_monopole(cl: FloatArray) -> None:
    """Tests that glass.grf.solve is auto differentiable when using JAX with monopole."""
    pytest.skip("glass.grf.solve is not auto differentiable")
    lam = 0.12345
    t = glass.grf.Lognormal(lam)
    cltol = 1e-7
    gl0 = 0.67891

    def solve_by_cl(cl):
        return glass.grf.solve(cl, t, monopole=gl0, cltol=cltol)

    jax.test_util.check_grads(
        solve_by_cl,
        (cl,),
        order=1,
    )

