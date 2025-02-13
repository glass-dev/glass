import numpy as np
import pytest

import glass.core.algorithm


def test_nnls(rng: np.random.Generator) -> None:
    # check output

    a = np.arange(25.0).reshape(-1, 5)
    b = np.arange(5.0)
    y = a @ b
    res = glass.core.algorithm.nnls(a, y)
    assert np.linalg.norm((a @ res) - y) < 1e-7

    a = rng.uniform(low=-10, high=10, size=[50, 10])
    b = np.abs(rng.uniform(low=-2, high=2, size=[10]))
    b[::2] = 0
    x = a @ b
    res = glass.core.algorithm.nnls(
        a, x, tol=500 * np.linalg.norm(a, 1) * np.spacing(1.0)
    )
    np.testing.assert_allclose(res, b, rtol=0.0, atol=1e-10)

    # check matrix and vector's shape

    a = rng.standard_normal((100, 20))
    b = rng.standard_normal((100,))

    with pytest.raises(ValueError, match="input `a` is not a matrix"):
        glass.core.algorithm.nnls(b, a)
    with pytest.raises(ValueError, match="input `b` is not a vector"):
        glass.core.algorithm.nnls(a, a)
    with pytest.raises(ValueError, match="the shapes of `a` and `b` do not match"):
        glass.core.algorithm.nnls(a.T, b)
