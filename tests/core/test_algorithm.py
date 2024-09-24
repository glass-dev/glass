import pytest


def test_nnls(rng):
    import numpy as np
    from scipy.optimize import nnls as nnls_scipy

    from glass.core.algorithm import nnls as nnls_glass

    a = rng.standard_normal((100, 20))
    b = rng.standard_normal((100,))

    x_glass = nnls_glass(a, b)
    x_scipy, _ = nnls_scipy(a, b)

    assert np.allclose(x_glass, x_scipy)

    with pytest.raises(ValueError, match="input `a` is not a matrix"):
        nnls_glass(b, a)
    with pytest.raises(ValueError, match="input `b` is not a vector"):
        nnls_glass(a, a)
    with pytest.raises(ValueError, match="the shapes of `a` and `b` do not match"):
        nnls_glass(a.T, b)
