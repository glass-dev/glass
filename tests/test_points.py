import numpy as np
import pytest
import pytest_mock

from glass.points import (
    effective_bias,
    linear_bias,
    loglinear_bias,
    position_weights,
    positions_from_delta,
    uniform_positions,
)


def catpos(pos):  # type: ignore[no-untyped-def]
    lon, lat, cnt = [], [], 0  # type: ignore[var-annotated]
    for lo, la, co in pos:
        lon = np.concatenate([lon, lo])  # type: ignore[assignment]
        lat = np.concatenate([lat, la])  # type: ignore[assignment]
        cnt = cnt + co
    return lon, lat, cnt


def test_effective_bias(mocker: pytest_mock.MockerFixture) -> None:
    # create a mock radial window function
    w = mocker.Mock()
    w.za = np.linspace(0, 2, 100)
    w.wa = np.full_like(w.za, 2.0)

    z = np.linspace(0, 1, 10)
    bz = np.zeros((10,))

    np.testing.assert_allclose(effective_bias(z, bz, w), np.zeros((10,)))  # type: ignore[no-untyped-call]

    z = np.zeros((10,))
    bz = np.full_like(z, 0.5)

    np.testing.assert_allclose(effective_bias(z, bz, w), np.zeros((10,)))  # type: ignore[no-untyped-call]

    z = np.linspace(0, 1, 10)
    bz = np.full_like(z, 0.5)

    np.testing.assert_allclose(effective_bias(z, bz, w), 0.25)  # type: ignore[no-untyped-call]


def test_linear_bias(rng: np.random.Generator) -> None:
    # test with 0 delta

    delta = np.zeros((2, 2))
    b = 2.0

    np.testing.assert_allclose(linear_bias(delta, b), np.zeros((2, 2)))  # type: ignore[no-untyped-call]

    # test with 0 b

    delta = rng.normal(5, 1, size=(2, 2))
    b = 0.0

    np.testing.assert_allclose(linear_bias(delta, b), np.zeros((2, 2)))  # type: ignore[no-untyped-call]

    # compare with original implementation

    delta = rng.normal(5, 1, size=(2, 2))
    b = 2.0

    np.testing.assert_allclose(linear_bias(delta, b), b * delta)  # type: ignore[no-untyped-call]


def test_loglinear_bias(rng: np.random.Generator) -> None:
    # test with 0 delta

    delta = np.zeros((2, 2))
    b = 2.0

    np.testing.assert_allclose(loglinear_bias(delta, b), np.zeros((2, 2)))  # type: ignore[no-untyped-call]

    # test with 0 b

    delta = rng.normal(5, 1, size=(2, 2))
    b = 0.0

    np.testing.assert_allclose(loglinear_bias(delta, b), np.zeros((2, 2)))  # type: ignore[no-untyped-call]

    # compare with numpy implementation

    delta = rng.normal(5, 1, size=(2, 2))
    b = 2.0

    np.testing.assert_allclose(loglinear_bias(delta, b), np.expm1(b * np.log1p(delta)))  # type: ignore[no-untyped-call]


def test_positions_from_delta(rng):  # type: ignore[no-untyped-def]  # noqa: PLR0915
    # case: single-dimensional input

    ngal = 1e-3
    delta = np.zeros(12)
    bias = 0.8
    vis = np.ones(12)

    lon, lat, cnt = catpos(positions_from_delta(ngal, delta, bias, vis))  # type: ignore[no-untyped-call]

    assert isinstance(cnt, int)
    assert lon.shape == lat.shape == (cnt,)

    # test with rng

    lon, lat, cnt = catpos(positions_from_delta(ngal, delta, bias, vis, rng=rng))  # type: ignore[no-untyped-call]

    assert isinstance(cnt, int)
    assert lon.shape == lat.shape == (cnt,)

    # case: Nons bias

    lon, lat, cnt = catpos(positions_from_delta(ngal, delta, None, vis))  # type: ignore[no-untyped-call]

    assert isinstance(cnt, int)
    assert lon.shape == lat.shape == (cnt,)

    # case: None vis

    lon, lat, cnt = catpos(positions_from_delta(ngal, delta, bias, None))  # type: ignore[no-untyped-call]

    assert isinstance(cnt, int)
    assert lon.shape == lat.shape == (cnt,)

    # case: remove monopole

    lon, lat, cnt = catpos(
        positions_from_delta(ngal, delta, bias, vis, remove_monopole=True)  # type: ignore[no-untyped-call]
    )

    assert isinstance(cnt, int)
    assert lon.shape == lat.shape == (cnt,)

    # case: negative delta

    lon, lat, cnt = catpos(
        positions_from_delta(ngal, np.linspace(-1, -1, 12), None, vis)  # type: ignore[no-untyped-call]
    )

    assert isinstance(cnt, int)
    assert lon == lat == []

    # case: large delta

    lon, lat, cnt = catpos(
        positions_from_delta(ngal, rng.normal(100, 1, size=(12,)), bias, vis)  # type: ignore[no-untyped-call]
    )

    assert isinstance(cnt, int)
    assert lon.shape == lat.shape == (cnt,)

    # case: multi-dimensional ngal

    ngal = [1e-3, 2e-3]  # type: ignore[assignment]
    delta = np.zeros(12)
    bias = 0.8
    vis = np.ones(12)

    lon, lat, cnt = catpos(positions_from_delta(ngal, delta, bias, vis))  # type: ignore[no-untyped-call]

    assert cnt.shape == (2,)
    assert lon.shape == (cnt.sum(),)
    assert lat.shape == (cnt.sum(),)

    # case: multi-dimensional delta

    ngal = 1e-3
    delta = np.zeros((3, 2, 12))
    bias = 0.8
    vis = np.ones(12)

    lon, lat, cnt = catpos(positions_from_delta(ngal, delta, bias, vis))  # type: ignore[no-untyped-call]

    assert cnt.shape == (3, 2)
    assert lon.shape == (cnt.sum(),)
    assert lat.shape == (cnt.sum(),)

    # case: multi-dimensional broadcasting

    ngal = [1e-3, 2e-3]  # type: ignore[assignment]
    delta = np.zeros((3, 1, 12))
    bias = 0.8
    vis = np.ones(12)

    lon, lat, cnt = catpos(positions_from_delta(ngal, delta, bias, vis))  # type: ignore[no-untyped-call]

    assert cnt.shape == (3, 2)
    assert lon.shape == (cnt.sum(),)
    assert lat.shape == (cnt.sum(),)

    # test TypeError

    with pytest.raises(TypeError, match="bias_model must be string or callable"):
        next(positions_from_delta(ngal, delta, bias, vis, bias_model=0))  # type: ignore[no-untyped-call]


def test_uniform_positions(rng):  # type: ignore[no-untyped-def]
    # case: scalar input

    ngal = 1e-3

    lon, lat, cnt = catpos(uniform_positions(ngal))  # type: ignore[no-untyped-call]

    # test with rng

    lon, lat, cnt = catpos(uniform_positions(ngal, rng=rng))  # type: ignore[no-untyped-call]

    assert isinstance(cnt, int)
    assert lon.shape == lat.shape == (cnt,)

    # case: 1-D array input

    ngal = [1e-3, 2e-3, 3e-3]  # type: ignore[assignment]

    lon, lat, cnt = catpos(uniform_positions(ngal))  # type: ignore[no-untyped-call]

    assert cnt.shape == (3,)
    assert lon.shape == lat.shape == (cnt.sum(),)

    # case: 2-D array input

    ngal = [[1e-3, 2e-3], [3e-3, 4e-3], [5e-3, 6e-3]]  # type: ignore[assignment]

    lon, lat, cnt = catpos(uniform_positions(ngal))  # type: ignore[no-untyped-call]

    assert cnt.shape == (3, 2)
    assert lon.shape == lat.shape == (cnt.sum(),)


def test_position_weights(rng):  # type: ignore[no-untyped-def]
    for bshape in None, (), (100,), (100, 1):
        for cshape in (100,), (100, 50), (100, 3, 2):
            counts = rng.random(cshape)
            bias = None if bshape is None else rng.random(bshape)

            weights = position_weights(counts, bias)  # type: ignore[no-untyped-call]

            expected = counts / counts.sum(axis=0, keepdims=True)
            if bias is not None:
                if np.ndim(bias) > np.ndim(expected):
                    expected = np.expand_dims(
                        expected,
                        tuple(range(np.ndim(expected), np.ndim(bias))),
                    )
                else:
                    bias = np.expand_dims(
                        bias,
                        tuple(range(np.ndim(bias), np.ndim(expected))),
                    )
                expected = bias * expected

            np.testing.assert_allclose(weights, expected)
