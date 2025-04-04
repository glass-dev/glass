import types

import numpy as np
import pytest
import tests.conftest

import glass.arraytools


@tests.conftest.array_api_compatible
def test_broadcast_first(xp: types.ModuleType) -> None:
    a = xp.ones((2, 3, 4))
    b = xp.ones((2, 1))

    # arrays with shape ((3, 4, 2)) and ((1, 2)) are passed
    # to xp.broadcast_arrays; hence it works
    a_a, b_a = glass.arraytools.broadcast_first(a, b)
    assert a_a.shape == (2, 3, 4)
    assert b_a.shape == (2, 3, 4)

    # plain xp.broadcast_arrays will not work
    with pytest.raises(ValueError, match="shape mismatch"):
        xp.broadcast_arrays(a, b)

    # arrays with shape ((5, 6, 4)) and ((6, 5)) are passed
    # to xp.broadcast_arrays; hence it will not work
    a = xp.ones((4, 5, 6))
    b = xp.ones((5, 6))

    with pytest.raises(ValueError, match="shape mismatch"):
        glass.arraytools.broadcast_first(a, b)

    # plain xp.broadcast_arrays will work
    a_a, b_a = xp.broadcast_arrays(a, b)

    assert a_a.shape == (4, 5, 6)
    assert b_a.shape == (4, 5, 6)


@tests.conftest.array_api_compatible
def test_broadcast_leading_axes(xp: types.ModuleType) -> None:
    a_in = 0
    b_in = xp.zeros((4, 10))
    c_in = xp.zeros((3, 1, 5, 6))

    dims, *rest = glass.arraytools.broadcast_leading_axes(
        (a_in, 0), (b_in, 1), (c_in, 2)
    )
    a_out, b_out, c_out = rest

    assert dims == (3, 4)
    assert a_out.shape == (3, 4)
    assert b_out.shape == (3, 4, 10)
    assert c_out.shape == (3, 4, 5, 6)


@tests.conftest.array_api_compatible
def test_ndinterp(xp: types.ModuleType) -> None:
    # test 1d interpolation

    xq = xp.asarray([0, 1, 2, 3, 4])
    yq = xp.asarray([1.1, 1.2, 1.3, 1.4, 1.5])

    x = 0.5
    y = glass.arraytools.ndinterp(x, xq, yq)
    assert xp.shape(y) == ()
    np.testing.assert_allclose(y, 1.15, atol=1e-15)

    x = xp.asarray([0.5, 1.5, 2.5])
    y = glass.arraytools.ndinterp(x, xq, yq)
    assert xp.shape(y) == (3,)
    np.testing.assert_allclose(y, [1.15, 1.25, 1.35], atol=1e-15)

    x = xp.asarray([[0.5, 1.5], [2.5, 3.5]])
    y = glass.arraytools.ndinterp(x, xq, yq)
    assert xp.shape(y) == (2, 2)
    np.testing.assert_allclose(y, [[1.15, 1.25], [1.35, 1.45]], atol=1e-15)

    # test nd interpolation in final axis

    yq = xp.asarray([[1.1, 1.2, 1.3, 1.4, 1.5], [2.1, 2.2, 2.3, 2.4, 2.5]])

    x = 0.5
    y = glass.arraytools.ndinterp(x, xq, yq)
    assert xp.shape(y) == (2,)
    np.testing.assert_allclose(y, [1.15, 2.15], atol=1e-15)

    x = xp.asarray([0.5, 1.5, 2.5])
    y = glass.arraytools.ndinterp(x, xq, yq)
    assert xp.shape(y) == (2, 3)
    np.testing.assert_allclose(y, [[1.15, 1.25, 1.35], [2.15, 2.25, 2.35]], atol=1e-15)

    x = xp.asarray([[0.5, 1.5], [2.5, 3.5]])
    y = glass.arraytools.ndinterp(x, xq, yq)
    assert xp.shape(y) == (2, 2, 2)
    np.testing.assert_allclose(
        y,
        [[[1.15, 1.25], [1.35, 1.45]], [[2.15, 2.25], [2.35, 2.45]]],
        atol=1e-15,
    )

    # test nd interpolation in middle axis

    yq = xp.asarray(
        [[[1.1], [1.2], [1.3], [1.4], [1.5]], [[2.1], [2.2], [2.3], [2.4], [2.5]]],
    )

    x = 0.5
    y = glass.arraytools.ndinterp(x, xq, yq, axis=1)
    assert xp.shape(y) == (2, 1)
    np.testing.assert_allclose(y, [[1.15], [2.15]], atol=1e-15)

    x = xp.asarray([0.5, 1.5, 2.5])
    y = glass.arraytools.ndinterp(x, xq, yq, axis=1)
    assert xp.shape(y) == (2, 3, 1)
    np.testing.assert_allclose(
        y,
        [[[1.15], [1.25], [1.35]], [[2.15], [2.25], [2.35]]],
        atol=1e-15,
    )

    x = xp.asarray([[0.5, 1.5, 2.5, 3.5], [3.5, 2.5, 1.5, 0.5], [0.5, 3.5, 1.5, 2.5]])
    y = glass.arraytools.ndinterp(x, xq, yq, axis=1)
    assert xp.shape(y) == (2, 3, 4, 1)
    np.testing.assert_allclose(
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


@tests.conftest.array_api_compatible
def test_trapezoid_product(xp: types.ModuleType) -> None:
    x1 = xp.linspace(0, 2, 100)
    f1 = xp.full_like(x1, 2.0)

    x2 = xp.linspace(1, 2, 10)
    f2 = xp.full_like(x2, 0.5)

    s = glass.arraytools.trapezoid_product((x1, f1), (x2, f2))

    np.testing.assert_allclose(s, 1.0)


@tests.conftest.array_api_compatible
def test_cumulative_trapezoid(xp: types.ModuleType) -> None:
    # 1D f and x

    f = xp.asarray([1, 2, 3, 4])
    x = xp.asarray([0, 1, 2, 3])

    # default dtype (int)

    ct = glass.arraytools.cumulative_trapezoid(f, x)
    np.testing.assert_allclose(ct, xp.asarray([0, 1, 4, 7]))

    # explicit dtype (float)

    ct = glass.arraytools.cumulative_trapezoid(f, x, dtype=float)
    np.testing.assert_allclose(ct, xp.asarray([0.0, 1.5, 4.0, 7.5]))

    # explicit return array

    out = xp.zeros((4,))
    ct = glass.arraytools.cumulative_trapezoid(f, x, dtype=float, out=out)
    np.testing.assert_equal(ct, out)

    # 2D f and 1D x

    f = xp.asarray([[1, 4, 9, 16], [2, 3, 5, 7]])
    x = xp.asarray([0, 1, 2.5, 4])

    # default dtype (int)

    ct = glass.arraytools.cumulative_trapezoid(f, x)
    np.testing.assert_allclose(ct, xp.asarray([[0, 2, 12, 31], [0, 2, 8, 17]]))

    # explicit dtype (float)

    ct = glass.arraytools.cumulative_trapezoid(f, x, dtype=float)
    np.testing.assert_allclose(
        ct, xp.asarray([[0.0, 2.5, 12.25, 31.0], [0.0, 2.5, 8.5, 17.5]])
    )

    # explicit return array

    out = xp.zeros((2, 4))
    ct = glass.arraytools.cumulative_trapezoid(f, x, dtype=float, out=out)
    np.testing.assert_equal(ct, out)
