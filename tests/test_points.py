from __future__ import annotations

import typing

import numpy as np
import numpy.typing as npt

from glass.points import position_weights, positions_from_delta, uniform_positions

if typing.TYPE_CHECKING:
    import collections.abc


def catpos(
    pos: collections.abc.Generator[
        tuple[
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            int | npt.NDArray[np.int_],
        ]
    ],
) -> tuple[
    list[float] | npt.NDArray[np.float64],
    list[float] | npt.NDArray[np.float64],
    int | npt.NDArray[np.int_],
]:
    lon: list[float] | npt.NDArray[np.float64] = []
    lat: list[float] | npt.NDArray[np.float64] = []
    cnt: int | npt.NDArray[np.int_] = 0
    for lo, la, co in pos:
        lon = np.concatenate([lon, lo])
        lat = np.concatenate([lat, la])
        cnt = cnt + co
    return lon, lat, cnt


def test_positions_from_delta() -> None:
    # case: single-dimensional input

    ngal: float | list[float] = 1e-3
    delta = np.zeros(12)
    bias = 0.8
    vis = np.ones(12)

    lon, lat, cnt = catpos(positions_from_delta(ngal, delta, bias, vis))

    assert isinstance(cnt, int)
    assert lon.shape == lat.shape == (cnt,)  # type: ignore[union-attr]

    # case: multi-dimensional ngal

    ngal = [1e-3, 2e-3]
    delta = np.zeros(12)
    bias = 0.8
    vis = np.ones(12)

    lon, lat, cnt = catpos(positions_from_delta(ngal, delta, bias, vis))

    assert cnt.shape == (2,)  # type: ignore[union-attr]
    assert lon.shape == (cnt.sum(),)  # type: ignore[union-attr]
    assert lat.shape == (cnt.sum(),)  # type: ignore[union-attr]

    # case: multi-dimensional delta

    ngal = 1e-3
    delta = np.zeros((3, 2, 12))
    bias = 0.8
    vis = np.ones(12)

    lon, lat, cnt = catpos(positions_from_delta(ngal, delta, bias, vis))

    assert cnt.shape == (3, 2)  # type: ignore[union-attr]
    assert lon.shape == (cnt.sum(),)  # type: ignore[union-attr]
    assert lat.shape == (cnt.sum(),)  # type: ignore[union-attr]

    # case: multi-dimensional broadcasting

    ngal = [1e-3, 2e-3]
    delta = np.zeros((3, 1, 12))
    bias = 0.8
    vis = np.ones(12)

    lon, lat, cnt = catpos(positions_from_delta(ngal, delta, bias, vis))

    assert cnt.shape == (3, 2)  # type: ignore[union-attr]
    assert lon.shape == (cnt.sum(),)  # type: ignore[union-attr]
    assert lat.shape == (cnt.sum(),)  # type: ignore[union-attr]


def test_uniform_positions() -> None:
    # case: scalar input

    ngal: float | list[float] | list[list[float]] = 1e-3

    lon, lat, cnt = catpos(uniform_positions(ngal))

    assert isinstance(cnt, int)
    assert lon.shape == lat.shape == (cnt,)  # type: ignore[union-attr]

    # case: 1-D array input

    ngal = [1e-3, 2e-3, 3e-3]

    lon, lat, cnt = catpos(uniform_positions(ngal))

    assert cnt.shape == (3,)  # type: ignore[union-attr]
    assert lon.shape == lat.shape == (cnt.sum(),)  # type: ignore[union-attr]

    # case: 2-D array input

    ngal = [[1e-3, 2e-3], [3e-3, 4e-3], [5e-3, 6e-3]]

    lon, lat, cnt = catpos(uniform_positions(ngal))

    assert cnt.shape == (3, 2)  # type: ignore[union-attr]
    assert lon.shape == lat.shape == (cnt.sum(),)  # type: ignore[union-attr]


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
