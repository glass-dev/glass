import numpy as np
import numpy.testing as npt


def broadcast_first():
    from glass.core.array import broadcast_first

    a = np.ones((2, 3, 4))
    b = np.ones((2, 1, 4))

    a_b, b_b = broadcast_first(a, b)

    assert a_b.shape == (2, 3, 4)
    assert b_b.shape == (2, 3, 4)

    a = np.ones((2, 2, 2))
    b = np.ones((2, 1))

    a_b, b_b = broadcast_first(a, b)

    assert a_b.shape == (2, 2, 2)
    assert b_b.shape == (2, 2, 2)


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


def test_cumtrapz():
    import numpy as np

    from glass.core.array import cumtrapz

    f = np.array([1, 2, 3, 4])
    x = np.array([0, 1, 2, 3])

    result = cumtrapz(f, x)
    assert np.allclose(result, np.array([0, 1, 4, 7]))

    result = cumtrapz(f, x, dtype=float)
    assert np.allclose(result, np.array([0.0, 1.5, 4.0, 7.5]))

    result = cumtrapz(f, x, dtype=float, out=np.zeros((4,)))
    assert np.allclose(result, np.array([0.0, 1.5, 4.0, 7.5]))

    f = np.array([[1, 4, 9, 16], [2, 3, 5, 7]])
    x = np.array([0, 1, 2.5, 4])

    result = cumtrapz(f, x)
    assert np.allclose(result, np.array([[[0, 2, 12, 31], [0, 2, 8, 17]]]))

    result = cumtrapz(f, x, dtype=float)
    assert np.allclose(
        result, np.array([[[0.0, 2.5, 12.25, 31.0], [0.0, 2.5, 8.5, 17.5]]])
    )

    result = cumtrapz(f, x, dtype=float, out=np.zeros((2, 4)))
    assert np.allclose(
        result, np.array([[[0.0, 2.5, 12.25, 31.0], [0.0, 2.5, 8.5, 17.5]]])
    )
