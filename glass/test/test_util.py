def test_tally():
    import numpy as np

    from glass.util import tally

    n = 100
    c = np.zeros(n)
    i = np.arange(4*n) % n

    tally(c, i)

    np.testing.assert_array_equal(c, 4)

    c = np.zeros(n)
    w = np.random.rand(4*n)

    tally(c, i, w)

    x = w.reshape(4, n).sum(axis=0)

    np.testing.assert_array_equal(c, x)


def test_accumulate():
    import numpy as np

    from glass.util import accumulate

    n = 100
    t = np.zeros(n)
    c = np.zeros(n)
    v = np.random.randn(4*n)
    i = np.arange(4*n) % n

    accumulate(t, c, v, i)

    x = v.reshape(4, n).sum(axis=0)

    np.testing.assert_array_equal(t, x)
    np.testing.assert_array_equal(c, 4)

    t = np.zeros(n)
    c = np.zeros(n)
    w = np.random.rand(4*n)

    accumulate(t, c, v, i, w)

    x = (w*v).reshape(4, n).sum(axis=0)
    y = w.reshape(4, n).sum(axis=0)

    np.testing.assert_array_equal(t, x)
    np.testing.assert_array_equal(c, y)
