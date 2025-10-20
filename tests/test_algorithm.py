from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

import glass.algorithm

if TYPE_CHECKING:
    import types

    import pytest_mock
    from conftest import UnifiedGenerator


def test_nnls(xp: types.ModuleType, urng: UnifiedGenerator) -> None:
    """Unit tests for glass.algorithm.nnls."""
    if xp.__name__ == "jax.numpy":
        pytest.skip("Arrays in nnls are not immutable, so do not support jax")

    # check output

    a = xp.reshape(xp.arange(25.0), (-1, 5))
    b = xp.arange(5.0)
    y = a @ b
    res = glass.algorithm.nnls(a, y)
    assert xp.linalg.vector_norm((a @ res) - y) < 1e-7

    a = urng.uniform(low=-10, high=10, size=[50, 10])
    b = xp.abs(urng.uniform(low=-2, high=2, size=[10]))
    b[::2] = 0
    x = a @ b
    res = glass.algorithm.nnls(
        a,
        x,
        tol=500 * xp.linalg.matrix_norm(a, ord=1) * xp.finfo(xp.float64).eps,
    )
    assert res == pytest.approx(b, rel=0.0, abs=1e-10)

    # check matrix and vector's shape

    a = urng.standard_normal((100, 20))
    b = urng.standard_normal((100,))

    with pytest.raises(ValueError, match="input `a` is not a matrix"):
        glass.algorithm.nnls(b, a)
    with pytest.raises(ValueError, match="input `b` is not a vector"):
        glass.algorithm.nnls(a, a)
    with pytest.raises(ValueError, match="the shapes of `a` and `b` do not match"):
        glass.algorithm.nnls(a.T, b)


def test_cov_clip(xp: types.ModuleType, urng: UnifiedGenerator) -> None:
    # prepare a random matrix
    m = urng.random((4, 4))

    # symmetric matrix
    a = (m + m.T) / 2

    # fix by clipping negative eigenvalues
    cov = glass.algorithm.cov_clip(a)

    # make sure all eigenvalues are positive
    assert xp.all(xp.linalg.eigvalsh(cov) >= 0)

    # fix by clipping negative eigenvalues
    cov = glass.algorithm.cov_clip(a, rtol=1.0)

    # make sure all eigenvalues are positive
    h = xp.max(xp.linalg.eigvalsh(a))
    np.testing.assert_allclose(xp.linalg.eigvalsh(cov), h)


def test_nearcorr(xp: types.ModuleType) -> None:
    # from Higham (2002)
    a = xp.asarray(
        [
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
    )
    b = xp.asarray(
        [
            [1.0000, 0.7607, 0.1573],
            [0.7607, 1.0000, 0.7607],
            [0.1573, 0.7607, 1.0000],
        ],
    )

    x = glass.algorithm.nearcorr(a)
    np.testing.assert_allclose(x, b, atol=0.0001)

    # explicit tolerance
    x = glass.algorithm.nearcorr(a, tol=1e-10)
    np.testing.assert_allclose(x, b, atol=0.0001)

    # no iterations
    x = glass.algorithm.nearcorr(a, niter=0)
    np.testing.assert_allclose(x, a)

    # non-square matrix should raise
    with pytest.raises(ValueError, match="non-square matrix"):
        glass.algorithm.nearcorr(xp.zeros((4, 3)))


def test_cov_nearest(
    xp: types.ModuleType,
    urng: UnifiedGenerator,
    mocker: pytest_mock.MockerFixture,
) -> None:
    # prepare a random matrix
    m = urng.random((4, 4))

    # symmetric matrix
    a = xp.eye(4) + (m + m.T) / 2

    # spy on the call to nearcorr
    nearcorr = mocker.spy(glass.algorithm, "nearcorr")

    # compute covariance
    cov = glass.algorithm.cov_nearest(a)

    # make sure all eigenvalues are positive
    assert xp.all(xp.linalg.eigvalsh(cov) >= 0)

    # get normalisation
    sq_d = xp.sqrt(xp.linalg.diagonal(a))
    norm = xp.linalg.outer(sq_d, sq_d)

    # make sure nearcorr was called with correct input
    nearcorr.assert_called_once()
    np.testing.assert_array_almost_equal_nulp(
        nearcorr.call_args_list[0].args[0],
        xp.divide(a, norm),
    )

    # cannot deal with negative variances
    with pytest.raises(ValueError, match="negative values"):
        glass.algorithm.cov_nearest(xp.asarray([[1, 0, 0], [0, 1, 0], [0, 0, -1]]))
