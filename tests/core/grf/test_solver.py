from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

import glass.grf

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from tests.conftest import Compare


@pytest.fixture(scope="session")
def cl() -> NDArray[np.float64]:
    lmax = 100
    ell = np.arange(lmax + 1)
    return 1e-2 / (2 * ell + 1) ** 2


def test_one_transformation(
    cl: NDArray[np.float64],
    compare: type[Compare],
    rng: np.random.Generator,
) -> None:
    lam = rng.random()
    t = glass.grf.Lognormal(lam)

    gl1, _, _ = glass.grf.solve(cl, t)
    gl2, _, _ = glass.grf.solve(cl, t, t)

    compare.assert_array_equal(gl1, gl2)


def test_pad(cl: NDArray[np.float64], rng: np.random.Generator) -> None:
    lam = rng.random()
    t = glass.grf.Lognormal(lam)

    # check that output size matches pad
    _, cl_out, _ = glass.grf.solve(cl, t, pad=2 * cl.size)

    assert cl_out.size == 3 * cl.size

    with pytest.raises(ValueError, match="pad must be a positive integer"):
        glass.grf.solve(cl, t, pad=-1)


def test_initial(
    cl: NDArray[np.float64],
    compare: type[Compare],
    rng: np.random.Generator,
) -> None:
    lam = rng.random()
    t = glass.grf.Lognormal(lam)

    gl = glass.grf.compute(cl, t)

    gl1, _, _ = glass.grf.solve(cl, t)
    gl2, _, _ = glass.grf.solve(cl, t, initial=gl)

    compare.assert_array_equal(gl1, gl2)


def test_no_iterations(cl: NDArray[np.float64], compare: type[Compare]) -> None:
    t = glass.grf.Lognormal()

    gl1 = glass.grf.compute(cl, t)
    gl2, _, _ = glass.grf.solve(cl, t, maxiter=0)

    compare.assert_array_equal(gl1, gl2)


def test_lognormal(
    cl: NDArray[np.float64],
    compare: type[Compare],
    rng: np.random.Generator,
) -> None:
    t1 = glass.grf.Lognormal()
    t2 = glass.grf.Lognormal()

    gl0 = rng.random()

    cltol = 1e-7

    gl, cl_, info = glass.grf.solve(cl, t1, t2, monopole=gl0, cltol=cltol)

    assert info > 0

    compare.assert_allclose(cl_[1 : cl.size], cl[1:], atol=0.0, rtol=cltol)

    gl_ = glass.grf.compute(cl_, t1, t2)

    assert gl[0] == gl0
    compare.assert_allclose(gl_[1 : gl.size], gl[1:])


def test_monopole(
    cl: NDArray[np.float64],
    compare: type[Compare],
    rng: np.random.Generator,
) -> None:
    t = glass.grf.Lognormal()

    cl[0] = rng.random()
    gl0 = rng.random()

    gl, cl_out, _ = glass.grf.solve(cl, t, monopole=None, gltol=1e-8)

    assert gl[0] != 0.0
    compare.assert_allclose(cl_out[0], cl[0])

    gl, cl_out, _ = glass.grf.solve(cl, t, monopole=gl0, gltol=1e-8)

    assert gl[0] == gl0
    with pytest.raises(AssertionError, match="Not equal to tolerance"):
        compare.assert_allclose(cl_out[0], cl[0])
