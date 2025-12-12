from __future__ import annotations

from typing import TYPE_CHECKING

import healpix
import numpy as np
import pytest

import glass

if TYPE_CHECKING:
    from types import ModuleType

    from numpy.typing import NDArray
    from pytest_mock import MockerFixture

    from glass._types import FloatArray, UnifiedGenerator
    from tests.fixtures.helper_classes import Compare, DataTransformer


def test_effective_bias(
    compare: type[Compare],
    mocker: MockerFixture,
    xp: ModuleType,
) -> None:
    # create a mock radial window function
    w = mocker.Mock()
    w.za = xp.linspace(0, 2, 100)
    w.wa = xp.full_like(w.za, 2.0)

    z = xp.linspace(0, 1, 10)
    bz = xp.zeros((10,))
    compare.assert_allclose(glass.effective_bias(z, bz, w), 0.0)

    z = xp.zeros((10,))
    bz = xp.full_like(z, 0.5)

    compare.assert_allclose(glass.effective_bias(z, bz, w), 0.0)

    z = xp.linspace(0, 1, 10)
    bz = xp.full_like(z, 0.5)

    compare.assert_allclose(glass.effective_bias(z, bz, w), 0.25)


def test_linear_bias(
    compare: type[Compare],
    urng: UnifiedGenerator,
    xp: ModuleType,
) -> None:
    # test with 0 delta

    delta = xp.zeros((2, 2))
    b = 2.0

    compare.assert_allclose(glass.linear_bias(delta, b), xp.zeros((2, 2)))

    # test with 0 b

    delta = urng.normal(5, 1, size=(2, 2))
    b = 0.0

    compare.assert_allclose(glass.linear_bias(delta, b), xp.zeros((2, 2)))

    # compare with original implementation

    delta = urng.normal(5, 1, size=(2, 2))
    b = 2.0

    compare.assert_allclose(glass.linear_bias(delta, b), b * delta)


def test_loglinear_bias(
    compare: type[Compare],
    urng: UnifiedGenerator,
    xp: ModuleType,
) -> None:
    # test with 0 delta

    delta = xp.zeros((2, 2))
    b = 2.0

    compare.assert_allclose(glass.loglinear_bias(delta, b), xp.zeros((2, 2)))

    # test with 0 b

    delta = urng.normal(5, 1, size=(2, 2))
    b = 0.0

    compare.assert_allclose(glass.loglinear_bias(delta, b), xp.zeros((2, 2)))

    # compare with numpy implementation

    delta = urng.normal(5, 1, size=(2, 2))
    b = 2.0

    compare.assert_allclose(
        glass.loglinear_bias(delta, b),
        xp.expm1(b * xp.log1p(delta)),
    )


def test_positions_from_delta(  # noqa: PLR0915
    compare: type[Compare],
    data_transformer: DataTransformer,
    rng: np.random.Generator,
) -> None:
    # create maps that saturate the batching in the function
    nside = 128
    npix = healpix.nside2npix(nside)

    # case: single-dimensional input

    ngal: float | NDArray[np.float64] = 1e-3
    delta = np.zeros(npix)
    bias = 0.8
    vis = np.ones(npix)

    lon, lat, cnt = data_transformer.catpos(
        glass.positions_from_delta(ngal, delta, bias, vis),
        xp=np,
    )

    assert isinstance(cnt, int)
    assert lon.shape == lat.shape == (cnt,)

    # test with rng

    lon, lat, cnt = data_transformer.catpos(
        glass.positions_from_delta(ngal, delta, bias, vis, rng=rng),
        xp=np,
    )

    assert isinstance(cnt, int)
    assert lon.shape == lat.shape == (cnt,)

    # case: Nons bias and callable bias model

    lon, lat, cnt = data_transformer.catpos(
        glass.positions_from_delta(ngal, delta, None, vis, bias_model=lambda x: x),
        xp=np,
    )

    assert isinstance(cnt, int)
    assert lon.shape == lat.shape == (cnt,)

    # case: None vis

    lon, lat, cnt = data_transformer.catpos(
        glass.positions_from_delta(ngal, delta, bias, None),
        xp=np,
    )

    assert isinstance(cnt, int)
    assert lon.shape == lat.shape == (cnt,)

    # case: remove monopole

    lon, lat, cnt = data_transformer.catpos(
        glass.positions_from_delta(ngal, delta, bias, vis, remove_monopole=True),
        xp=np,
    )

    assert isinstance(cnt, int)
    assert lon.shape == lat.shape == (cnt,)

    # case: negative delta

    lon, lat, cnt = data_transformer.catpos(
        glass.positions_from_delta(ngal, np.linspace(-1, -1, npix), None, vis),
        xp=np,
    )

    assert isinstance(cnt, int)
    compare.assert_allclose(lon, [])
    compare.assert_allclose(lat, [])

    # case: large delta

    lon, lat, cnt = data_transformer.catpos(
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

    lon, lat, cnt = data_transformer.catpos(
        glass.positions_from_delta(ngal, delta, bias, vis),
        xp=np,
    )

    assert isinstance(cnt, np.ndarray)
    assert cnt.shape == (2,)
    assert lon.shape == (cnt.sum(),)
    assert lat.shape == (cnt.sum(),)

    # case: multi-dimensional delta

    ngal = 1e-3
    delta = np.zeros((3, 2, 12))
    bias = 0.8
    vis = np.ones(12)

    lon, lat, cnt = data_transformer.catpos(
        glass.positions_from_delta(ngal, delta, bias, vis),
        xp=np,
    )

    assert isinstance(cnt, np.ndarray)
    assert cnt.shape == (3, 2)
    assert lon.shape == (cnt.sum(),)
    assert lat.shape == (cnt.sum(),)

    # case: multi-dimensional broadcasting

    ngal = np.array([1e-3, 2e-3])
    delta = np.zeros((3, 1, 12))
    bias = 0.8
    vis = np.ones(12)

    lon, lat, cnt = data_transformer.catpos(
        glass.positions_from_delta(ngal, delta, bias, vis),
        xp=np,
    )

    assert isinstance(cnt, np.ndarray)
    assert cnt.shape == (3, 2)
    assert lon.shape == (cnt.sum(),)
    assert lat.shape == (cnt.sum(),)

    # case: only the southern hemisphere is visible

    vis[: vis.size // 2] = 0.0

    lon, lat, cnt = data_transformer.catpos(
        glass.positions_from_delta(ngal, delta, bias, vis),
        xp=np,
    )

    assert isinstance(cnt, np.ndarray)
    assert cnt.shape == (3, 2)
    assert lon.shape == (cnt.sum(),)
    assert lat.shape == (cnt.sum(),)

    # test TypeError

    with pytest.raises(TypeError, match="bias_model must be string or callable"):
        next(glass.positions_from_delta(ngal, delta, bias, vis, bias_model=0))


def test_uniform_positions(
    data_transformer: DataTransformer,
    urng: UnifiedGenerator,
    xp: ModuleType,
) -> None:
    if xp.__name__ == "jax.numpy":
        pytest.skip(
            "Arrays in uniform_positions are not immutable, so do not support jax",
        )

    # case: scalar input

    ngal: float | FloatArray = 1e-3

    lon, lat, cnt = data_transformer.catpos(glass.uniform_positions(ngal, xp=xp), xp=xp)

    # Pass non-arrays without xp

    with pytest.raises(TypeError, match="Unrecognized array input"):
        next(glass.uniform_positions(ngal))

    # test with rng

    lon, lat, cnt = data_transformer.catpos(
        glass.uniform_positions(ngal, rng=urng, xp=xp),
        xp=xp,
    )

    assert isinstance(cnt, int)
    assert lon.shape == lat.shape == (cnt,)

    # case: 1-D array input

    ngal = xp.asarray([1e-3, 2e-3, 3e-3])

    lon, lat, cnt = data_transformer.catpos(glass.uniform_positions(ngal), xp=xp)

    assert not isinstance(cnt, int)
    assert cnt.__array_namespace__() == xp
    assert cnt.shape == (3,)
    assert lon.shape == lat.shape == (xp.sum(cnt),)

    # case: 2-D array input

    ngal = xp.asarray([[1e-3, 2e-3], [3e-3, 4e-3], [5e-3, 6e-3]])

    lon, lat, cnt = data_transformer.catpos(glass.uniform_positions(ngal), xp=xp)

    assert not isinstance(cnt, int)
    assert cnt.__array_namespace__() == xp
    assert cnt.shape == (3, 2)
    assert lon.shape == lat.shape == (xp.sum(cnt),)


def test_position_weights(
    compare: type[Compare],
    urng: UnifiedGenerator,
    xp: ModuleType,
) -> None:
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

            compare.assert_allclose(weights, expected)


def test_displace_arg_complex(compare: type[Compare], xp: ModuleType) -> None:
    """Test displace function with complex-valued displacement."""
    d = 5.0  # deg
    r = d / 180 * xp.pi

    # displace the origin so everything is easy
    lon0 = xp.asarray(0.0)
    lat0 = xp.asarray(0.0)

    # north
    lon, lat = glass.displace(lon0, lat0, xp.asarray(r + 0j))
    compare.assert_allclose([lon, lat], [0.0, d])

    # south
    lon, lat = glass.displace(lon0, lat0, xp.asarray(-r + 0j))
    compare.assert_allclose([lon, lat], [0.0, -d], atol=1e-15)

    # east
    lon, lat = glass.displace(lon0, lat0, xp.asarray(1j * r))
    compare.assert_allclose([lon, lat], [-d, 0.0], atol=1e-15)

    # west
    lon, lat = glass.displace(lon0, lat0, xp.asarray(-1j * r))
    compare.assert_allclose([lon, lat], [d, 0.0], atol=1e-15)


def test_displace_arg_real(compare: type[Compare], xp: ModuleType) -> None:
    """Test displace function with real-valued argument."""
    d = 5.0  # deg
    r = d / 180 * xp.pi

    # displace the origin so everything is easy
    lon0 = xp.asarray(0.0)
    lat0 = xp.asarray(0.0)

    # north
    lon, lat = glass.displace(lon0, lat0, xp.asarray([r, 0]))
    compare.assert_allclose([lon, lat], [0.0, d])

    # south
    lon, lat = glass.displace(lon0, lat0, xp.asarray([-r, 0]))
    compare.assert_allclose([lon, lat], [0.0, -d], atol=1e-15)

    # east
    lon, lat = glass.displace(lon0, lat0, xp.asarray([0, r]))
    compare.assert_allclose([lon, lat], [-d, 0.0], atol=1e-15)

    # west
    lon, lat = glass.displace(lon0, lat0, xp.asarray([0, -r]))
    compare.assert_allclose([lon, lat], [d, 0.0], atol=1e-15)


def test_displace_abs(
    compare: type[Compare],
    urng: UnifiedGenerator,
    xp: ModuleType,
) -> None:
    """Check that points are displaced by the correct angular distance."""
    n = 1_000
    abs_alpha = urng.uniform(0, 2 * xp.pi, size=n)
    arg_alpha = urng.uniform(-xp.pi, xp.pi, size=n)

    lon_ = urng.uniform(-xp.pi, xp.pi, size=n) / xp.pi * 180
    lat_ = xp.asin(urng.uniform(-1, 1, size=n)) / xp.pi * 180

    lon, lat = glass.displace(lon_, lat_, abs_alpha * xp.exp(1j * arg_alpha))

    th = (90.0 - lat) / 180 * xp.pi
    th_ = (90.0 - lat_) / 180 * xp.pi
    delta= (lon - lon_) / 180 * xp.pi

    cos_a = xp.cos(th) * xp.cos(th_) + xp.cos(delta) * xp.sin(th) * xp.sin(th_)

    compare.assert_allclose(cos_a, xp.cos(abs_alpha))


def test_displacement(
    compare: type[Compare],
    urng: UnifiedGenerator,
    xp: ModuleType,
) -> None:
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
        compare.assert_allclose(alpha_, alpha)

    # test on an array
    alpha = glass.displacement(
        urng.uniform(-180.0, 180.0, size=(20, 1)),
        urng.uniform(-90.0, 90.0, size=(20, 1)),
        urng.uniform(-180.0, 180.0, size=5),
        urng.uniform(-90.0, 90.0, size=5),
    )
    assert alpha.shape == (20, 5)
