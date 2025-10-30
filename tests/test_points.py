from __future__ import annotations

from typing import TYPE_CHECKING

import healpix
import numpy as np
import pytest

import glass

if TYPE_CHECKING:
    from collections.abc import Generator
    from types import ModuleType

    import pytest_mock
    from numpy.typing import NDArray

    from glass._types import FloatArray, IntArray, UnifiedGenerator


def catpos(
    pos: Generator[
        tuple[
            FloatArray,
            FloatArray,
            IntArray,
        ]
    ],
    *,
    xp: ModuleType,
) -> tuple[
    FloatArray,
    FloatArray,
    IntArray,
]:
    lon = xp.empty(0)
    lat = xp.empty(0)
    cnt: IntArray = 0
    for lo, la, co in pos:
        lon = xp.concat([lon, lo])
        lat = xp.concat([lat, la])
        cnt = cnt + co
    return lon, lat, cnt


def test_effective_bias(xp: ModuleType, mocker: pytest_mock.MockerFixture) -> None:
    # create a mock radial window function
    w = mocker.Mock()
    w.za = xp.linspace(0, 2, 100)
    w.wa = xp.full_like(w.za, 2.0)

    z = xp.linspace(0, 1, 10)
    bz = xp.zeros((10,))
    np.testing.assert_allclose(glass.effective_bias(z, bz, w), 0.0)

    z = xp.zeros((10,))
    bz = xp.full_like(z, 0.5)

    np.testing.assert_allclose(glass.effective_bias(z, bz, w), 0.0)

    z = xp.linspace(0, 1, 10)
    bz = xp.full_like(z, 0.5)

    np.testing.assert_allclose(glass.effective_bias(z, bz, w), 0.25)


def test_linear_bias(xp: ModuleType, urng: UnifiedGenerator) -> None:
    # test with 0 delta

    delta = xp.zeros((2, 2))
    b = 2.0

    np.testing.assert_allclose(glass.linear_bias(delta, b), xp.zeros((2, 2)))

    # test with 0 b

    delta = urng.normal(5, 1, size=(2, 2))
    b = 0.0

    np.testing.assert_allclose(glass.linear_bias(delta, b), xp.zeros((2, 2)))

    # compare with original implementation

    delta = urng.normal(5, 1, size=(2, 2))
    b = 2.0

    np.testing.assert_allclose(glass.linear_bias(delta, b), b * delta)


def test_loglinear_bias(xp: ModuleType, urng: UnifiedGenerator) -> None:
    # test with 0 delta

    delta = xp.zeros((2, 2))
    b = 2.0

    np.testing.assert_allclose(glass.loglinear_bias(delta, b), xp.zeros((2, 2)))

    # test with 0 b

    delta = urng.normal(5, 1, size=(2, 2))
    b = 0.0

    np.testing.assert_allclose(glass.loglinear_bias(delta, b), xp.zeros((2, 2)))

    # compare with numpy implementation

    delta = urng.normal(5, 1, size=(2, 2))
    b = 2.0

    np.testing.assert_allclose(
        glass.loglinear_bias(delta, b),
        xp.expm1(b * xp.log1p(delta)),
    )


def test_positions_from_delta(rng: np.random.Generator) -> None:  # noqa: PLR0915
    # create maps that saturate the batching in the function
    nside = 128
    npix = healpix.nside2npix(nside)

    # case: single-dimensional input

    ngal: float | NDArray[np.float64] = 1e-3
    delta = np.zeros(npix)
    bias = 0.8
    vis = np.ones(npix)

    lon, lat, cnt = catpos(glass.positions_from_delta(ngal, delta, bias, vis), xp=np)

    assert isinstance(cnt, int)
    assert lon.shape == lat.shape == (cnt,)

    # test with rng

    lon, lat, cnt = catpos(
        glass.positions_from_delta(ngal, delta, bias, vis, rng=rng), xp=np
    )

    assert isinstance(cnt, int)
    assert lon.shape == lat.shape == (cnt,)

    # case: Nons bias and callable bias model

    lon, lat, cnt = catpos(
        glass.positions_from_delta(ngal, delta, None, vis, bias_model=lambda x: x),
        xp=np,
    )

    assert isinstance(cnt, int)
    assert lon.shape == lat.shape == (cnt,)

    # case: None vis

    lon, lat, cnt = catpos(glass.positions_from_delta(ngal, delta, bias, None), xp=np)

    assert isinstance(cnt, int)
    assert lon.shape == lat.shape == (cnt,)

    # case: remove monopole

    lon, lat, cnt = catpos(
        glass.positions_from_delta(ngal, delta, bias, vis, remove_monopole=True), xp=np
    )

    assert isinstance(cnt, int)
    assert lon.shape == lat.shape == (cnt,)

    # case: negative delta

    lon, lat, cnt = catpos(
        glass.positions_from_delta(ngal, np.linspace(-1, -1, npix), None, vis), xp=np
    )

    assert isinstance(cnt, int)
    np.testing.assert_allclose(lon, [])
    np.testing.assert_allclose(lat, [])

    # case: large delta

    lon, lat, cnt = catpos(
        glass.positions_from_delta(ngal, rng.normal(100, 1, size=(npix,)), bias, vis),
        xp=np,
    )

    assert isinstance(cnt, int)
    assert lon.shape == lat.shape == (cnt,)

    # case: multi-dimensional ngal

    ngal = np.array([1e-3, 2e-3])
    delta = np.zeros(12)
    bias = 0.8
    vis = np.ones(12)

    lon, lat, cnt = catpos(glass.positions_from_delta(ngal, delta, bias, vis), xp=np)

    assert isinstance(cnt, np.ndarray)
    assert cnt.shape == (2,)
    assert lon.shape == (cnt.sum(),)
    assert lat.shape == (cnt.sum(),)

    # case: multi-dimensional delta

    ngal = 1e-3
    delta = np.zeros((3, 2, 12))
    bias = 0.8
    vis = np.ones(12)

    lon, lat, cnt = catpos(glass.positions_from_delta(ngal, delta, bias, vis), xp=np)

    assert isinstance(cnt, np.ndarray)
    assert cnt.shape == (3, 2)
    assert lon.shape == (cnt.sum(),)
    assert lat.shape == (cnt.sum(),)

    # case: multi-dimensional broadcasting

    ngal = np.array([1e-3, 2e-3])
    delta = np.zeros((3, 1, 12))
    bias = 0.8
    vis = np.ones(12)

    lon, lat, cnt = catpos(glass.positions_from_delta(ngal, delta, bias, vis), xp=np)

    assert isinstance(cnt, np.ndarray)
    assert cnt.shape == (3, 2)
    assert lon.shape == (cnt.sum(),)
    assert lat.shape == (cnt.sum(),)

    # case: only the southern hemisphere is visible

    vis[: vis.size // 2] = 0.0

    lon, lat, cnt = catpos(glass.positions_from_delta(ngal, delta, bias, vis), xp=np)

    assert isinstance(cnt, np.ndarray)
    assert cnt.shape == (3, 2)
    assert lon.shape == (cnt.sum(),)
    assert lat.shape == (cnt.sum(),)

    # test TypeError

    with pytest.raises(TypeError, match="bias_model must be string or callable"):
        next(glass.positions_from_delta(ngal, delta, bias, vis, bias_model=0))


def test_uniform_positions(xp: ModuleType, urng: UnifiedGenerator) -> None:
    if xp.__name__ == "jax.numpy":
        pytest.skip(
            "Arrays in uniform_positions are not immutable, so do not support jax"
        )

    # case: scalar input

    ngal: float | FloatArray = 1e-3

    lon, lat, cnt = catpos(glass.uniform_positions(ngal, xp=xp), xp=xp)

    # Pass non-arrays without xp

    with pytest.raises(TypeError, match="Unrecognized array input"):
        next(glass.uniform_positions(ngal))

    # test with rng

    lon, lat, cnt = catpos(glass.uniform_positions(ngal, rng=urng, xp=xp), xp=xp)

    assert isinstance(cnt, int)
    assert lon.shape == lat.shape == (cnt,)

    # case: 1-D array input

    ngal = xp.asarray([1e-3, 2e-3, 3e-3])

    lon, lat, cnt = catpos(glass.uniform_positions(ngal), xp=xp)

    assert not isinstance(cnt, int)
    assert cnt.__array_namespace__() == xp
    assert cnt.shape == (3,)
    assert lon.shape == lat.shape == (xp.sum(cnt),)

    # case: 2-D array input

    ngal = xp.asarray([[1e-3, 2e-3], [3e-3, 4e-3], [5e-3, 6e-3]])

    lon, lat, cnt = catpos(glass.uniform_positions(ngal), xp=xp)

    assert not isinstance(cnt, int)
    assert cnt.__array_namespace__() == xp
    assert cnt.shape == (3, 2)
    assert lon.shape == lat.shape == (xp.sum(cnt),)


def test_position_weights(xp: ModuleType, urng: UnifiedGenerator) -> None:
    """Unit tests for glass.points.position_weights."""
    for bshape in None, (), (100,), (100, 1):
        for cshape in (100,), (100, 50), (100, 3, 2):
            counts = urng.random(cshape)
            bias = None if bshape is None else urng.random(bshape)

            weights = glass.position_weights(counts, bias)

            expected = counts / xp.sum(counts, axis=0, keepdims=True)
            if bias is not None:
                if bias.ndim > expected.ndim:
                    expected = xp.expand_dims(
                        expected,
                        axis=tuple(range(expected.ndim, bias.ndim)),
                    )
                else:
                    bias = xp.expand_dims(
                        bias,
                        axis=tuple(range(bias.ndim, expected.ndim)),
                    )
                expected = bias * expected

            np.testing.assert_allclose(weights, expected)


def test_displace_arg_complex(xp: ModuleType) -> None:
    """Test displace function with complex-valued displacement."""
    d = 5.0  # deg
    r = d / 180 * xp.pi

    # displace the origin so everything is easy
    lon0 = xp.asarray(0.0)
    lat0 = xp.asarray(0.0)

    # north
    lon, lat = glass.displace(lon0, lat0, xp.asarray(r + 0j))
    assert np.allclose([lon, lat], [0.0, d])

    # south
    lon, lat = glass.displace(lon0, lat0, xp.asarray(-r + 0j))
    assert np.allclose([lon, lat], [0.0, -d])

    # east
    lon, lat = glass.displace(lon0, lat0, xp.asarray(1j * r))
    assert np.allclose([lon, lat], [-d, 0.0])

    # west
    lon, lat = glass.displace(lon0, lat0, xp.asarray(-1j * r))
    assert np.allclose([lon, lat], [d, 0.0])


def test_displace_arg_real(xp: ModuleType) -> None:
    """Test displace function with real-valued argument."""
    d = 5.0  # deg
    r = d / 180 * xp.pi

    # displace the origin so everything is easy
    lon0 = xp.asarray(0.0)
    lat0 = xp.asarray(0.0)

    # north
    lon, lat = glass.displace(lon0, lat0, xp.asarray([r, 0]))
    assert np.allclose([lon, lat], [0.0, d])

    # south
    lon, lat = glass.displace(lon0, lat0, xp.asarray([-r, 0]))
    assert np.allclose([lon, lat], [0.0, -d])

    # east
    lon, lat = glass.displace(lon0, lat0, xp.asarray([0, r]))
    assert np.allclose([lon, lat], [-d, 0.0])

    # west
    lon, lat = glass.displace(lon0, lat0, xp.asarray([0, -r]))
    assert np.allclose([lon, lat], [d, 0.0])


def test_displace_abs(xp: ModuleType, urng: UnifiedGenerator) -> None:
    """Check that points are displaced by the correct angular distance."""
    n = 1000
    abs_alpha = urng.uniform(0, 2 * xp.pi, size=n)
    arg_alpha = urng.uniform(-xp.pi, xp.pi, size=n)

    lon_ = urng.uniform(-np.pi, np.pi, size=n) / xp.pi * 180
    lat_ = xp.asin(urng.uniform(-1, 1, size=n)) / xp.pi * 180

    lon, lat = glass.displace(lon_, lat_, abs_alpha * xp.exp(1j * arg_alpha))

    th = (90.0 - lat) / 180 * xp.pi
    th_ = (90.0 - lat_) / 180 * xp.pi
    delt = (lon - lon_) / 180 * xp.pi

    cos_a = xp.cos(th) * xp.cos(th_) + xp.cos(delt) * xp.sin(th) * xp.sin(th_)

    assert np.allclose(cos_a, xp.cos(abs_alpha))


def test_displacement(xp: ModuleType, urng: UnifiedGenerator) -> None:
    """Check that displacement of points is computed correctly."""
    # unit changes for displacements
    deg5 = xp.asarray(5.0) / 180 * xp.pi
    north = xp.exp(xp.asarray(1j * 0.0))
    east = xp.exp(xp.asarray(1j * (xp.pi / 2)))
    south = xp.exp(xp.asarray(1j * xp.pi))
    west = xp.exp(xp.asarray(1j * (3 * xp.pi / 2)))

    zero = xp.asarray(0.0)
    five = xp.asarray(5.0)
    ninety = xp.asarray(90.0)

    # test data: coordinates and expected displacement
    data = [
        # equator
        (zero, zero, zero, five, deg5 * north),
        (zero, zero, -five, zero, deg5 * east),
        (zero, zero, zero, -five, deg5 * south),
        (zero, zero, five, zero, deg5 * west),
        # pole
        (zero, ninety, ninety * 2, ninety - five, deg5 * north),
        (zero, ninety, -ninety, ninety - five, deg5 * east),
        (zero, ninety, zero, ninety - five, deg5 * south),
        (zero, ninety, ninety, ninety - five, deg5 * west),
    ]

    # test each displacement individually
    for from_lon, from_lat, to_lon, to_lat, alpha in data:
        alpha_ = glass.displacement(from_lon, from_lat, to_lon, to_lat)
        assert np.allclose(alpha_, alpha), (
            f"displacement from ({from_lon}, {from_lat}) to ({to_lon}, {to_lat})"
            f"\ndistance: expected {xp.abs(alpha)}, got {xp.abs(alpha_)}"
            f"\ndirection: expected {xp.angle(alpha)}, got {xp.angle(alpha_)}"
        )

    # test on an array
    alpha = glass.displacement(
        urng.uniform(-180.0, 180.0, size=(20, 1)),
        urng.uniform(-90.0, 90.0, size=(20, 1)),
        urng.uniform(-180.0, 180.0, size=5),
        urng.uniform(-90.0, 90.0, size=5),
    )
    assert alpha.shape == (20, 5)
