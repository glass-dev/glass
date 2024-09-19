import pytest

# check if scipy is available for testing
try:
    import scipy
except ImportError:
    HAVE_SCIPY = False
else:
    del scipy
    HAVE_SCIPY = True


@pytest.fixture
def rng():
    import numpy as np

    return np.random.default_rng(seed=42)


@pytest.mark.skipif(not HAVE_SCIPY, reason="test requires SciPy")
def test_nnls(rng):
    import numpy as np
    from scipy.optimize import nnls as nnls_scipy

    from glass.core.algorithm import nnls as nnls_glass

    a = rng.standard_normal((100, 20))
    b = rng.standard_normal((100,))

    x_glass = nnls_glass(a, b)
    x_scipy, _ = nnls_scipy(a, b)

    np.testing.assert_allclose(x_glass, x_scipy)
