import numpy as np
import numpy.testing as npt
import pytest

# check if scipy is available for testing
try:
    import scipy
except ImportError:
    HAVE_SCIPY = False
else:
    del scipy
    HAVE_SCIPY = True


def broadcast_first():
    from glass.core.array import broadcast_first

    a = np.ones((2, 3, 4))
    b = np.ones((2, 1))

    # arrays with shape ((3, 4, 2)) and ((1, 2)) are passed
    # to np.broadcast_arrays; hence it works
    a_a, b_a = broadcast_first(a, b)
    assert a_a.shape == (2, 3, 4)
    assert b_a.shape == (2, 3, 4)

    # plain np.broadcast_arrays will not work
    with pytest.raises(ValueError, match="shape mismatch"):
        np.broadcast_arrays(a, b)

    # arrays with shape ((5, 6, 4)) and ((6, 5)) are passed
    # to np.broadcast_arrays; hence it will not work
    a = np.ones((4, 5, 6))
    b = np.ones((5, 6))

    with pytest.raises(ValueError, match="shape mismatch"):
        broadcast_first(a, b)

    # plain np.broadcast_arrays will work
    a_a, b_a = broadcast_first(a, b)

    assert a_a.shape == (4, 5, 6)
    assert b_a.shape == (4, 5, 6)


def test_broadcast_leading_axes():
    from glass.core.array import broadcast_leading_axes

    a = 0
    b = np.zeros((4, 10))
    c = np.zeros((3, 1, 5, 6))

    dims, a, b, c = broadcast_leading_axes((a, 0), (b, 1), (c, 2))

    assert dims == (3, 4)
    assert a.shape == (3, 4)
    assert b.shape == (3, 4, 10)
    assert c.shape == (3, 4, 5, 6)


def test_ndinterp():
    from glass.core.array import ndinterp

    # test 1d interpolation

    xp = [0, 1, 2, 3, 4]
    yp = [1.1, 1.2, 1.3, 1.4, 1.5]

    x = 0.5
    y = ndinterp(x, xp, yp)
    assert np.shape(y) == ()
    npt.assert_allclose(y, 1.15, atol=1e-15)

    x = [0.5, 1.5, 2.5]
    y = ndinterp(x, xp, yp)
    assert np.shape(y) == (3,)
    npt.assert_allclose(y, [1.15, 1.25, 1.35], atol=1e-15)

    x = [[0.5, 1.5], [2.5, 3.5]]
    y = ndinterp(x, xp, yp)
    assert np.shape(y) == (2, 2)
    npt.assert_allclose(y, [[1.15, 1.25], [1.35, 1.45]], atol=1e-15)

    # test nd interpolation in final axis

    yp = [[1.1, 1.2, 1.3, 1.4, 1.5], [2.1, 2.2, 2.3, 2.4, 2.5]]

    x = 0.5
    y = ndinterp(x, xp, yp)
    assert np.shape(y) == (2,)
    npt.assert_allclose(y, [1.15, 2.15], atol=1e-15)

    x = [0.5, 1.5, 2.5]
    y = ndinterp(x, xp, yp)
    assert np.shape(y) == (2, 3)
    npt.assert_allclose(y, [[1.15, 1.25, 1.35], [2.15, 2.25, 2.35]], atol=1e-15)

    x = [[0.5, 1.5], [2.5, 3.5]]
    y = ndinterp(x, xp, yp)
    assert np.shape(y) == (2, 2, 2)
    npt.assert_allclose(
        y,
        [[[1.15, 1.25], [1.35, 1.45]], [[2.15, 2.25], [2.35, 2.45]]],
        atol=1e-15,
    )

    # test nd interpolation in middle axis

    yp = [[[1.1], [1.2], [1.3], [1.4], [1.5]], [[2.1], [2.2], [2.3], [2.4], [2.5]]]

    x = 0.5
    y = ndinterp(x, xp, yp, axis=1)
    assert np.shape(y) == (2, 1)
    npt.assert_allclose(y, [[1.15], [2.15]], atol=1e-15)

    x = [0.5, 1.5, 2.5]
    y = ndinterp(x, xp, yp, axis=1)
    assert np.shape(y) == (2, 3, 1)
    npt.assert_allclose(
        y,
        [[[1.15], [1.25], [1.35]], [[2.15], [2.25], [2.35]]],
        atol=1e-15,
    )

    x = [[0.5, 1.5, 2.5, 3.5], [3.5, 2.5, 1.5, 0.5], [0.5, 3.5, 1.5, 2.5]]
    y = ndinterp(x, xp, yp, axis=1)
    assert np.shape(y) == (2, 3, 4, 1)
    npt.assert_allclose(
        y,
        [
            [
                [[1.15], [1.25], [1.35], [1.45]],
                [[1.45], [1.35], [1.25], [1.15]],
                [[1.15], [1.45], [1.25], [1.35]],
            ],
            [
                [[2.15], [2.25], [2.35], [2.45]],
                [[2.45], [2.35], [2.25], [2.15]],
                [[2.15], [2.45], [2.25], [2.35]],
            ],
        ],
        atol=1e-15,
    )


def test_trapz_product():
    from glass.core.array import trapz_product

    x1 = np.linspace(0, 2, 100)
    f1 = np.full_like(x1, 2.0)

    x2 = np.linspace(1, 2, 10)
    f2 = np.full_like(x2, 0.5)

    s = trapz_product((x1, f1), (x2, f2))

    assert np.allclose(s, 1.0)


@pytest.mark.skipif(not HAVE_SCIPY, reason="test requires SciPy")
def test_cumtrapz():
    from scipy.integrate import cumulative_trapezoid

    from glass.core.array import cumtrapz

    # 1D f and x

    f = np.array([1, 2, 3, 4])
    x = np.array([0, 1, 2, 3])

    # default dtype (int - not supported by scipy)

    glass_ct = cumtrapz(f, x)
    npt.assert_allclose(glass_ct, np.array([0, 1, 4, 7]))

    # explicit dtype (float)

    glass_ct = cumtrapz(f, x, dtype=float)
    scipy_ct = cumulative_trapezoid(f, x, initial=0)
    npt.assert_allclose(glass_ct, scipy_ct)

    # explicit return array

    result = cumtrapz(f, x, dtype=float, out=np.zeros((4,)))
    scipy_ct = cumulative_trapezoid(f, x, initial=0)
    npt.assert_allclose(result, scipy_ct)

    # 2D f and 1D x

    f = np.array([[1, 4, 9, 16], [2, 3, 5, 7]])
    x = np.array([0, 1, 2.5, 4])

    # default dtype (int - not supported by scipy)

    glass_ct = cumtrapz(f, x)
    npt.assert_allclose(glass_ct, np.array([[0, 2, 12, 31], [0, 2, 8, 17]]))

    # explicit dtype (float)

    glass_ct = cumtrapz(f, x, dtype=float)
    scipy_ct = cumulative_trapezoid(f, x, initial=0)
    npt.assert_allclose(glass_ct, scipy_ct)

    # explicit return array

    glass_ct = cumtrapz(f, x, dtype=float, out=np.zeros((2, 4)))
    scipy_ct = cumulative_trapezoid(f, x, initial=0)
    npt.assert_allclose(glass_ct, scipy_ct)
