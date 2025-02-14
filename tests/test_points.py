from __future__ import annotations

from typing import TYPE_CHECKING

import healpix
import numpy as np
import pytest

from glass import (
    effective_bias,
    linear_bias,
    loglinear_bias,
    position_weights,
    positions_from_delta,
    uniform_positions,
)

if TYPE_CHECKING:
    from collections.abc import Generator

    import pytest_mock
    from numpy.typing import NDArray


def catpos(
    pos: Generator[
        tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            int | NDArray[np.int_],
        ]
    ],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    int | NDArray[np.int_],
]:
    lon = np.empty(0)
    lat = np.empty(0)
    cnt: int | NDArray[np.int_] = 0
    for lo, la, co in pos:
        lon = np.concatenate([lon, lo])
        lat = np.concatenate([lat, la])
        cnt = cnt + co
    return lon, lat, cnt


def test_effective_bias(mocker: pytest_mock.MockerFixture) -> None:
    # create a mock radial window function
    w = mocker.Mock()
    w.za = np.linspace(0, 2, 100)
    w.wa = np.full_like(w.za, 2.0)

    z = np.linspace(0, 1, 10)
    bz = np.zeros((10,))

    np.testing.assert_allclose(effective_bias(z, bz, w), np.zeros((10,)))

    z = np.zeros((10,))
    bz = np.full_like(z, 0.5)

    np.testing.assert_allclose(effective_bias(z, bz, w), np.zeros((10,)))

    z = np.linspace(0, 1, 10)
    bz = np.full_like(z, 0.5)

    np.testing.assert_allclose(effective_bias(z, bz, w), 0.25)


def test_linear_bias(rng: np.random.Generator) -> None:
    # test with 0 delta

    delta = np.zeros((2, 2))
    b = 2.0

    np.testing.assert_allclose(linear_bias(delta, b), np.zeros((2, 2)))

    # test with 0 b

    delta = rng.normal(5, 1, size=(2, 2))
    b = 0.0

    np.testing.assert_allclose(linear_bias(delta, b), np.zeros((2, 2)))

    # compare with original implementation

    delta = rng.normal(5, 1, size=(2, 2))
    b = 2.0

    np.testing.assert_allclose(linear_bias(delta, b), b * delta)


def test_loglinear_bias(rng: np.random.Generator) -> None:
    # test with 0 delta

    delta = np.zeros((2, 2))
    b = 2.0

    np.testing.assert_allclose(loglinear_bias(delta, b), np.zeros((2, 2)))

    # test with 0 b

    delta = rng.normal(5, 1, size=(2, 2))
    b = 0.0

    np.testing.assert_allclose(loglinear_bias(delta, b), np.zeros((2, 2)))

    # compare with numpy implementation

    delta = rng.normal(5, 1, size=(2, 2))
    b = 2.0

    np.testing.assert_allclose(loglinear_bias(delta, b), np.expm1(b * np.log1p(delta)))


def test_positions_from_delta(rng: np.random.Generator) -> None:  # noqa: PLR0915
    # create maps that saturate the batching in the function
    nside = 128
    npix = healpix.nside2npix(nside)

    # case: single-dimensional input

    ngal: float | NDArray[np.float64] = 1e-3
    delta = np.zeros(npix)
    bias = 0.8
    vis = np.ones(npix)

    lon, lat, cnt = catpos(positions_from_delta(ngal, delta, bias, vis))

    assert isinstance(cnt, int)
    assert lon.shape == lat.shape == (cnt,)

    # test with rng

    lon, lat, cnt = catpos(positions_from_delta(ngal, delta, bias, vis, rng=rng))

    assert isinstance(cnt, int)
    assert lon.shape == lat.shape == (cnt,)

    # case: Nons bias and callable bias model

    lon, lat, cnt = catpos(
        positions_from_delta(ngal, delta, None, vis, bias_model=lambda x: x)
    )

    assert isinstance(cnt, int)
    assert lon.shape == lat.shape == (cnt,)

    # case: None vis

    lon, lat, cnt = catpos(positions_from_delta(ngal, delta, bias, None))

    assert isinstance(cnt, int)
    assert lon.shape == lat.shape == (cnt,)

    # case: remove monopole

    lon, lat, cnt = catpos(
        positions_from_delta(ngal, delta, bias, vis, remove_monopole=True)
    )

    assert isinstance(cnt, int)
    assert lon.shape == lat.shape == (cnt,)

    # case: negative delta

    lon, lat, cnt = catpos(
        positions_from_delta(ngal, np.linspace(-1, -1, npix), None, vis)
    )

    assert isinstance(cnt, int)
    np.testing.assert_allclose(lon, [])
    np.testing.assert_allclose(lat, [])

    # case: large delta

    lon, lat, cnt = catpos(
        positions_from_delta(ngal, rng.normal(100, 1, size=(npix,)), bias, vis)
    )

    assert isinstance(cnt, int)
    assert lon.shape == lat.shape == (cnt,)

    # case: multi-dimensional ngal

    ngal = np.array([1e-3, 2e-3])
    delta = np.zeros(12)
    bias = 0.8
    vis = np.ones(12)

    lon, lat, cnt = catpos(positions_from_delta(ngal, delta, bias, vis))

    assert isinstance(cnt, np.ndarray)
    assert cnt.shape == (2,)
    assert lon.shape == (cnt.sum(),)
    assert lat.shape == (cnt.sum(),)

    # case: multi-dimensional delta

    ngal = 1e-3
    delta = np.zeros((3, 2, 12))
    bias = 0.8
    vis = np.ones(12)

    lon, lat, cnt = catpos(positions_from_delta(ngal, delta, bias, vis))

    assert isinstance(cnt, np.ndarray)
    assert cnt.shape == (3, 2)
    assert lon.shape == (cnt.sum(),)
    assert lat.shape == (cnt.sum(),)

    # case: multi-dimensional broadcasting

    ngal = np.array([1e-3, 2e-3])
    delta = np.zeros((3, 1, 12))
    bias = 0.8
    vis = np.ones(12)

    lon, lat, cnt = catpos(positions_from_delta(ngal, delta, bias, vis))

    assert isinstance(cnt, np.ndarray)
    assert cnt.shape == (3, 2)
    assert lon.shape == (cnt.sum(),)
    assert lat.shape == (cnt.sum(),)

    # case: only the southern hemisphere is visible

    vis[: vis.size // 2] = 0.0

    lon, lat, cnt = catpos(positions_from_delta(ngal, delta, bias, vis))

    assert isinstance(cnt, np.ndarray)
    assert cnt.shape == (3, 2)
    assert lon.shape == (cnt.sum(),)
    assert lat.shape == (cnt.sum(),)

    # test TypeError

    with pytest.raises(TypeError, match="bias_model must be string or callable"):
        next(positions_from_delta(ngal, delta, bias, vis, bias_model=0))  # type: ignore[arg-type]


def test_uniform_positions(rng: np.random.Generator) -> None:
    # case: scalar input

    ngal: float | NDArray[np.float64] = 1e-3

    lon, lat, cnt = catpos(uniform_positions(ngal))

    # test with rng

    lon, lat, cnt = catpos(uniform_positions(ngal, rng=rng))

    assert isinstance(cnt, int)
    assert lon.shape == lat.shape == (cnt,)

    # case: 1-D array input

    ngal = np.array([1e-3, 2e-3, 3e-3])

    lon, lat, cnt = catpos(uniform_positions(ngal))

    assert isinstance(cnt, np.ndarray)
    assert cnt.shape == (3,)
    assert lon.shape == lat.shape == (cnt.sum(),)

    # case: 2-D array input

    ngal = np.array([[1e-3, 2e-3], [3e-3, 4e-3], [5e-3, 6e-3]])

    lon, lat, cnt = catpos(uniform_positions(ngal))

    assert isinstance(cnt, np.ndarray)
    assert cnt.shape == (3, 2)
    assert lon.shape == lat.shape == (cnt.sum(),)


def test_position_weights(rng: np.random.Generator) -> None:
    for bshape in None, (), (100,), (100, 1):
        for cshape in (100,), (100, 50), (100, 3, 2):
            counts = rng.random(cshape)
            bias = None if bshape is None else rng.random(bshape)

            weights = position_weights(counts, bias)

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
