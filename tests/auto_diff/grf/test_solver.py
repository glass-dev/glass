from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

jax = pytest.importorskip("jax", reason="tests require jax")
import jax.numpy as jnp  # noqa: E402

import glass  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Callable

    from glass._types import AnyArray, FloatArray


@pytest.fixture(scope="session")
def cl(get_cl: Callable[..., FloatArray]) -> FloatArray:
    """Generate a consistent cl for JAX."""
    return get_cl(jnp)


def test_one_transformation(cl: FloatArray) -> None:
    """Tests glass.grf.solve is auto differentiable when using JAX."""
    pytest.skip("glass.grf.solve is not auto differentiable")
    lam = 0.12345
    t = glass.grf.Lognormal(lam)

    def solve_by_cl(cl: AnyArray) -> AnyArray:
        return glass.grf.solve(cl, t)

    jax.test_util.check_grads(
        solve_by_cl,
        (cl,),
        order=1,
    )


def test_pad(cl: FloatArray) -> None:
    """Tests glass.grf.solve is auto differentiable when using JAX and pad."""
    pytest.skip("glass.grf.solve is not auto differentiable")
    lam = 0.12345
    t = glass.grf.Lognormal(lam)

    def solve_by_cl(cl: AnyArray) -> AnyArray:
        return glass.grf.solve(cl, t, pad=2 * cl.shape[0])

    jax.test_util.check_grads(
        solve_by_cl,
        (cl,),
        order=1,
    )


def test_initial(cl: FloatArray) -> None:
    """Tests glass.grf.solve is auto differentiable when using JAX and initial."""
    pytest.skip("glass.grf.solve is not auto differentiable")
    lam = 0.12345
    t = glass.grf.Lognormal(lam)
    gl = glass.grf.compute(cl, t)

    def solve_by_cl(cl: AnyArray) -> AnyArray:
        return glass.grf.solve(cl, t, initial=gl)

    jax.test_util.check_grads(
        solve_by_cl,
        (cl,),
        order=1,
    )


def test_no_iterations(cl: FloatArray) -> None:
    """Tests glass.grf.solve is auto differentiable when using JAX and maxiter=0."""
    pytest.skip("glass.grf.solve is not auto differentiable")
    lam = 0.12345
    t = glass.grf.Lognormal(lam)

    def solve_by_cl(cl: AnyArray) -> AnyArray:
        return glass.grf.solve(cl, t, maxiter=0)

    jax.test_util.check_grads(
        solve_by_cl,
        (cl,),
        order=1,
    )


def test_monopole(cl: FloatArray) -> None:
    """Tests glass.grf.solve is auto differentiable when using JAX with monopole."""
    pytest.skip("glass.grf.solve is not auto differentiable")
    lam = 0.12345
    t = glass.grf.Lognormal(lam)
    cltol = 1e-7
    gl0 = 0.67891

    def solve_by_cl(cl: AnyArray) -> AnyArray:
        return glass.grf.solve(cl, t, monopole=gl0, cltol=cltol)

    jax.test_util.check_grads(
        solve_by_cl,
        (cl,),
        order=1,
    )
