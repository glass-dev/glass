import pytest

# check if scipy is available for testing
try:
    import scipy
except ImportError:
    HAVE_SCIPY = False
else:
    del scipy
    HAVE_SCIPY = True


@pytest.mark.skipif(not HAVE_SCIPY, reason="test requires SciPy")
def test_nnls():
    import numpy as np
    from scipy.optimize import nnls as nnls_scipy
    from glass.core.algorithm import nnls as nnls_glass

    a = np.random.randn(100, 20)
    b = np.random.randn(100)

    x_glass = nnls_glass(a, b)
    x_scipy, _ = nnls_scipy(a, b)

    np.testing.assert_allclose(x_glass, x_scipy)
