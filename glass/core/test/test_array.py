import numpy as np
import numpy.testing as npt


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
    npt.assert_allclose(y, [[1.15, 1.25],
                            [1.35, 1.45]], atol=1e-15)

    # test nd interpolation in final axis

    yp = [[1.1, 1.2, 1.3, 1.4, 1.5],
          [2.1, 2.2, 2.3, 2.4, 2.5]]

    x = 0.5
    y = ndinterp(x, xp, yp)
    assert np.shape(y) == (2,)
    npt.assert_allclose(y, [1.15, 2.15], atol=1e-15)

    x = [0.5, 1.5, 2.5]
    y = ndinterp(x, xp, yp)
    assert np.shape(y) == (2, 3)
    npt.assert_allclose(y, [[1.15, 1.25, 1.35],
                            [2.15, 2.25, 2.35]], atol=1e-15)

    x = [[0.5, 1.5], [2.5, 3.5]]
    y = ndinterp(x, xp, yp)
    assert np.shape(y) == (2, 2, 2)
    npt.assert_allclose(y, [[[1.15, 1.25],
                             [1.35, 1.45]],
                            [[2.15, 2.25],
                             [2.35, 2.45]]], atol=1e-15)

    # test nd interpolation in middle axis

    yp = [[[1.1], [1.2], [1.3], [1.4], [1.5]],
          [[2.1], [2.2], [2.3], [2.4], [2.5]]]

    x = 0.5
    y = ndinterp(x, xp, yp, axis=1)
    assert np.shape(y) == (2, 1)
    npt.assert_allclose(y, [[1.15], [2.15]], atol=1e-15)

    x = [0.5, 1.5, 2.5]
    y = ndinterp(x, xp, yp, axis=1)
    assert np.shape(y) == (2, 3, 1)
    npt.assert_allclose(y, [[[1.15], [1.25], [1.35]],
                            [[2.15], [2.25], [2.35]]], atol=1e-15)

    x = [[0.5, 1.5, 2.5, 3.5], [3.5, 2.5, 1.5, 0.5], [0.5, 3.5, 1.5, 2.5]]
    y = ndinterp(x, xp, yp, axis=1)
    assert np.shape(y) == (2, 3, 4, 1)
    npt.assert_allclose(y, [[[[1.15], [1.25], [1.35], [1.45]],
                             [[1.45], [1.35], [1.25], [1.15]],
                             [[1.15], [1.45], [1.25], [1.35]]],
                            [[[2.15], [2.25], [2.35], [2.45]],
                             [[2.45], [2.35], [2.25], [2.15]],
                             [[2.15], [2.45], [2.25], [2.35]]]], atol=1e-15)
