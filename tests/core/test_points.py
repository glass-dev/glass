from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest

import array_api_extra as xpx

import glass
import glass.healpix as hp
from glass._array_api_utils import xp_additions as uxpx

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType
    from typing import Any

    from pytest_mock import MockerFixture

    from glass._types import UnifiedGenerator
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


def test_broadcast_inputs(
    compare: type[Compare],
    xp: ModuleType,
) -> None:
    bias_in = 0.8
    delta_in = xp.zeros((3, 1, 12))
    ngal_in = xp.asarray([1e-3, 2e-3])
    vis_in = xp.repeat(xp.asarray([0.0, 1.0]), 6)

    bias, delta, dims, ngal, vis = glass.points._broadcast_inputs(
        bias_in,
        delta_in,
        ngal_in,
        vis_in,
    )

    assert dims == (3, 2)
    assert bias.shape == dims  # ty: ignore[possibly-missing-attribute]
    assert xp.all(bias == bias_in)
    compare.assert_array_equal(delta, xp.zeros_like(delta))
    assert ngal.shape == dims  # ty: ignore[possibly-missing-attribute]
    compare.assert_array_equal(ngal[0, :], ngal_in)  # ty: ignore[not-subscriptable]
    assert vis.shape == delta.shape  # ty: ignore[possibly-missing-attribute]
    compare.assert_array_equal(vis[0, 0, :], vis_in)  # ty: ignore[not-subscriptable]


@pytest.mark.parametrize(
    "bias_model",
    [
        glass.linear_bias,
        glass.loglinear_bias,
    ],
)
def test_compute_density_contrast(
    bias_model: Callable[..., Any],
    compare: type[Compare],
    xp: ModuleType,
) -> None:
    bias = 0.8 * xp.ones((3, 2))
    bias_model = glass.linear_bias
    delta = xp.zeros((3, 2, 12))
    k = (1, 1)

    n = glass.points._compute_density_contrast(
        bias,
        bias_model,
        delta,
        k,
    )

    assert n.shape[0] == delta.shape[-1]
    compare.assert_array_equal(n, xp.zeros_like(n))


@pytest.mark.parametrize("remove_monopole", [False, True])
def test_compute_expected_count(
    xp: ModuleType,
    *,
    remove_monopole: bool,
) -> None:
    k = (1, 1)
    n_in = xp.zeros(12)
    ngal = xp.tile(xp.asarray([1e-3, 2e-3]), (3, 1))

    n = glass.points._compute_expected_count(
        k,
        n_in,
        ngal,
        remove_monopole=remove_monopole,
    )

    assert n.shape == n_in.shape
    assert xp.all(n == n[0])


def test_apply_visibility(
    compare: type[Compare],
    xp: ModuleType,
) -> None:
    k = (1, 1)
    n_in = 24751.77674965 * xp.ones(12)
    vis = xp.tile(xp.repeat(xp.asarray([0.0, 1.0]), 6), (3, 2, 1))

    n = glass.points._apply_visibility(
        k,
        xp.asarray(n_in, copy=True),
        vis,
    )

    compare.assert_array_equal(n[:6], xp.zeros_like(n[:6]))
    compare.assert_array_equal(n[6:], n_in[6:])


def test_sample_number_galaxies(
    compare: type[Compare],
    xp: ModuleType,
) -> None:
    n_in = xp.repeat(xp.asarray([0.0, 24751.77674965]), 6)

    n = glass.points._sample_number_galaxies(n_in)

    compare.assert_array_equal(n[:6], xp.zeros_like(n[:6]))
    compare.assert_allclose(n[6:], n_in[6:], atol=250)


def test_sample_number_galaxies_rng(
    compare: type[Compare],
    urng: UnifiedGenerator,
    xp: ModuleType,
) -> None:
    n_in = xp.repeat(xp.asarray([0.0, 24751.77674965]), 6)

    n = glass.points._sample_number_galaxies(n_in, rng=urng)

    compare.assert_array_equal(n[:6], xp.zeros_like(n[:6]))
    compare.assert_allclose(n[6:], n_in[6:], atol=250)


def test_sample_galaxies_per_pixel(
    data_transformer: DataTransformer,
    xp: ModuleType,
) -> None:
    batch = 1000000
    dims = (3, 2)
    k = (1, 1)
    n = xp.asarray([0, 0, 0, 0, 0, 0, 24885, 24945, 24505, 24877, 24546, 24693])

    lon, lat, count = data_transformer.catpos(
        glass.points._sample_galaxies_per_pixel(batch, dims, k, n),
        xp=xp,
    )

    assert count.shape == dims
    assert count[k] == lon.shape[0]
    assert count[k] == lat.shape[0]
    assert tuple(int(i[0]) for i in xp.nonzero(count)) == k
    assert xp.min(lon) >= 0
    assert xp.max(lon) < 360
    assert xp.min(lat) >= -90
    assert xp.max(lat) <= 90


def test_positions_from_delta(  # noqa: PLR0915
    compare: type[Compare],
    data_transformer: DataTransformer,
    urng: UnifiedGenerator,
    xp: ModuleType,
) -> None:
    # create maps that saturate the batching in the function
    nside = 128
    npix = hp.nside2npix(nside)

    # case: single-dimensional input

    ngal = 1e-3
    delta = xp.zeros(npix)
    bias = 0.8
    vis = xp.ones(npix)

    lon, lat, count = data_transformer.catpos(
        glass.positions_from_delta(ngal, delta, bias, vis),
        xp=xp,
    )

    assert int(count) == count
    assert lon.shape == lat.shape == (count,)

    # test with rng

    lon, lat, count = data_transformer.catpos(
        glass.positions_from_delta(ngal, delta, bias, vis, rng=urng),
        xp=xp,
    )

    assert int(count) == count
    assert lon.shape == lat.shape == (count,)

    # case: Nons bias and callable bias model

    lon, lat, count = data_transformer.catpos(
        glass.positions_from_delta(ngal, delta, None, vis, bias_model=lambda x: x),
        xp=xp,
    )

    assert int(count) == count
    assert lon.shape == lat.shape == (count,)

    # case: None vis

    lon, lat, count = data_transformer.catpos(
        glass.positions_from_delta(ngal, delta, bias, None),
        xp=xp,
    )

    assert int(count) == count
    assert lon.shape == lat.shape == (count,)

    # case: remove monopole

    lon, lat, count = data_transformer.catpos(
        glass.positions_from_delta(ngal, delta, bias, vis, remove_monopole=True),
        xp=xp,
    )

    assert int(count) == count
    assert lon.shape == lat.shape == (count,)

    # case: negative delta

    lon, lat, count = data_transformer.catpos(
        glass.positions_from_delta(ngal, xp.linspace(-1, -1, npix), None, vis),
        xp=xp,
    )

    assert int(count) == count
    compare.assert_allclose(lon, [])
    compare.assert_allclose(lat, [])

    # case: large delta

    lon, lat, count = data_transformer.catpos(
        glass.positions_from_delta(ngal, urng.normal(100, 1, size=(npix,)), bias, vis),
        xp=xp,
    )

    assert int(count) == count
    assert lon.shape == lat.shape == (count,)

    # case: multi-dimensional ngal

    ngal = xp.asarray([1e-3, 2e-3])
    delta = xp.zeros(12)
    bias = 0.8
    vis = xp.ones(12)

    lon, lat, count = data_transformer.catpos(
        glass.positions_from_delta(ngal, delta, bias, vis),
        xp=xp,
    )

    assert count.shape == (2,)
    assert lon.shape == (xp.sum(count),)
    assert lat.shape == (xp.sum(count),)

    # case: multi-dimensional delta

    ngal = 1e-3
    delta = xp.zeros((3, 2, 12))
    bias = 0.8
    vis = xp.ones(12)

    lon, lat, count = data_transformer.catpos(
        glass.positions_from_delta(ngal, delta, bias, vis),
        xp=xp,
    )

    assert count.shape == (3, 2)
    assert lon.shape == (xp.sum(count),)
    assert lat.shape == (xp.sum(count),)

    # case: multi-dimensional broadcasting

    ngal = xp.asarray([1e-3, 2e-3])
    delta = xp.zeros((3, 1, 12))
    bias = 0.8
    vis = xp.ones(12)

    lon, lat, count = data_transformer.catpos(
        glass.positions_from_delta(ngal, delta, bias, vis),
        xp=xp,
    )

    assert count.shape == (3, 2)
    assert lon.shape == (xp.sum(count),)
    assert lat.shape == (xp.sum(count),)

    # case: only the southern hemisphere is visible

    vis = xpx.at(vis)[: vis.shape[0] // 2].set(0.0)

    lon, lat, count = data_transformer.catpos(
        glass.positions_from_delta(ngal, delta, bias, vis),
        xp=xp,
    )

    assert count.shape == (3, 2)
    assert lon.shape == (xp.sum(count),)
    assert lat.shape == (xp.sum(count),)

    # test TypeError

    with pytest.raises(TypeError, match="bias_model must be callable"):
        next(glass.positions_from_delta(ngal, delta, bias, vis, bias_model=0))


def test_uniform_positions(
    data_transformer: DataTransformer,
    urng: UnifiedGenerator,
    xp: ModuleType,
) -> None:
    # case: scalar input

    ngal = 1e-3

    lon, lat, count = data_transformer.catpos(
        glass.uniform_positions(ngal, xp=xp),
        xp=xp,
    )

    # Pass non-arrays without xp

    with pytest.raises(
        TypeError,
        match="array_namespace requires at least one non-scalar array input",
    ):
        next(glass.uniform_positions(ngal))

    # test with rng

    lon, lat, count = data_transformer.catpos(
        glass.uniform_positions(ngal, rng=urng, xp=xp),
        xp=xp,
    )

    assert int(count) == count
    assert lon.shape == lat.shape == (count,)

    # case: 1-D array input

    ngal = xp.asarray([1e-3, 2e-3, 3e-3])

    lon, lat, count = data_transformer.catpos(glass.uniform_positions(ngal), xp=xp)

    assert count.__array_namespace__() == xp
    assert count.shape == (3,)
    assert lon.shape == lat.shape == (xp.sum(count),)

    # case: 2-D array input

    ngal = xp.asarray([[1e-3, 2e-3], [3e-3, 4e-3], [5e-3, 6e-3]])

    lon, lat, count = data_transformer.catpos(glass.uniform_positions(ngal), xp=xp)

    assert count.__array_namespace__() == xp
    assert count.shape == (3, 2)
    assert lon.shape == lat.shape == (xp.sum(count),)


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
    r = d / 180 * math.pi

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
    compare.assert_allclose([lon, lat], [d, 0.0], atol=1e-15)

    # west
    lon, lat = glass.displace(lon0, lat0, xp.asarray(-1j * r))
    compare.assert_allclose([lon, lat], [-d, 0.0], atol=1e-15)


def test_displace_arg_real(compare: type[Compare], xp: ModuleType) -> None:
    """Test displace function with real-valued argument."""
    d = 5.0  # deg
    r = d / 180 * math.pi

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
    compare.assert_allclose([lon, lat], [d, 0.0], atol=1e-15)

    # west
    lon, lat = glass.displace(lon0, lat0, xp.asarray([0, -r]))
    compare.assert_allclose([lon, lat], [-d, 0.0], atol=1e-15)


def test_displace_abs(
    compare: type[Compare],
    urng: UnifiedGenerator,
    xp: ModuleType,
) -> None:
    """Check that points are displaced by the correct angular distance."""
    n = 1_000
    abs_alpha = urng.uniform(0, 2 * math.pi, size=n)
    arg_alpha = urng.uniform(-math.pi, math.pi, size=n)

    lon_ = urng.uniform(-math.pi, math.pi, size=n) / math.pi * 180
    lat_ = xp.asin(urng.uniform(-1, 1, size=n)) / math.pi * 180

    lon, lat = glass.displace(lon_, lat_, abs_alpha * xp.exp(1j * arg_alpha))

    th = (90.0 - lat) / 180 * math.pi
    th_ = (90.0 - lat_) / 180 * math.pi
    delta = (lon - lon_) / 180 * math.pi

    cos_a = xp.cos(th) * xp.cos(th_) + xp.cos(delta) * xp.sin(th) * xp.sin(th_)

    compare.assert_allclose(cos_a, xp.cos(abs_alpha))


def test_displacement(
    compare: type[Compare],
    urng: UnifiedGenerator,
    xp: ModuleType,
) -> None:
    """Check that displacement of points is computed correctly."""
    # unit changes for displacements
    deg5 = xp.asarray(5.0) / 180 * math.pi
    north = xp.exp(xp.asarray(1j * 0.0))
    east = xp.exp(xp.asarray(1j * (math.pi / 2)))
    south = xp.exp(xp.asarray(1j * math.pi))
    west = xp.exp(xp.asarray(1j * (3 * math.pi / 2)))

    zero = xp.asarray(0.0)
    five = xp.asarray(5.0)
    ninety = xp.asarray(90.0)

    # test data: coordinates and expected displacement
    data = [
        # equator
        (zero, zero, zero, five, deg5 * north),
        (zero, zero, five, zero, deg5 * east),
        (zero, zero, zero, -five, deg5 * south),
        (zero, zero, -five, zero, deg5 * west),
        # pole
        (zero, ninety, ninety * 2, ninety - five, deg5 * north),
        (zero, ninety, ninety, ninety - five, deg5 * east),
        (zero, ninety, zero, ninety - five, deg5 * south),
        (zero, ninety, -ninety, ninety - five, deg5 * west),
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


def test_displacement_zerodist(
    compare: type[Compare],
    urng: UnifiedGenerator,
    xp: ModuleType,
) -> None:
    """Check that zero displacement is computed correctly."""
    lon = urng.uniform(-180.0, 180.0, size=100)
    lat = urng.uniform(-90.0, 90.0, size=100)

    compare.assert_allclose(
        glass.displacement(lon, lat, lon, lat),
        xp.zeros(100),
    )


def test_displacement_consistent(
    compare: type[Compare],
    urng: UnifiedGenerator,
    xp: ModuleType,
) -> None:
    """Check displacement is consistent with displace."""
    n = 1_000

    # magnitude and angle of displacement we want to achieve
    r = xp.acos(urng.uniform(-1.0, 1.0, size=n))
    x = urng.uniform(-math.pi, math.pi, size=n)

    # displace at random positions on the sphere
    from_lon = urng.uniform(-180.0, 180.0, size=n)
    from_lat = xp.asin(urng.uniform(-1.0, 1.0, size=n)) / math.pi * 180.0

    # compute the intended displacement
    alpha_in = r * xp.exp(1j * x)

    # displace random points
    to_lon, to_lat = glass.displace(from_lon, from_lat, alpha_in)

    # measure displacement
    alpha_out = glass.displacement(from_lon, from_lat, to_lon, to_lat)

    compare.assert_allclose(alpha_out, alpha_in, atol=0.0, rtol=1e-10)


def test_displacement_random(
    compare: type[Compare],
    urng: UnifiedGenerator,
    xp: ModuleType,
) -> None:
    """Check displacement for random points."""
    n = 1_000

    # magnitude and angle of displacement we want to achieve
    r = xp.acos(urng.uniform(-1.0, 1.0, size=n))
    x = urng.uniform(-math.pi, math.pi, size=n)

    # displacement at random positions on the sphere
    theta = xp.acos(urng.uniform(-1.0, 1.0, size=n))
    phi = urng.uniform(-math.pi, math.pi, size=n)

    # rotation matrix that moves (0, 0, 1) to theta and phi
    zero = xp.zeros(n)
    one = xp.ones(n)
    rot_y = xp.stack(
        [
            xp.cos(theta), zero, xp.sin(theta),
            zero, one, zero,
            -xp.sin(theta), zero, xp.cos(theta),
        ],
        axis=1,
    )  # fmt: skip
    rot_z = xp.stack(
        [
            xp.cos(phi), -xp.sin(phi), zero,
            xp.sin(phi), xp.cos(phi), zero,
            zero, zero, one,
        ],
        axis=1,
    )  # fmt: skip
    rot = xp.reshape(rot_z, (n, 3, 3)) @ xp.reshape(rot_y, (n, 3, 3))

    # meta-check that rotation works by rotating (0, 0, 1) to theta and phi
    u = xp.stack(
        [
            xp.sin(theta) * xp.cos(phi),
            xp.sin(theta) * xp.sin(phi),
            xp.cos(theta),
        ],
        axis=1,
    )
    compare.assert_allclose(rot @ xp.asarray([0.0, 0.0, 1.0]), u)

    # meta-check that recovering theta and phi from vector works
    compare.assert_allclose(xp.atan2(xp.hypot(u[:, 0], u[:, 1]), u[:, 2]), theta)
    compare.assert_allclose(xp.atan2(u[:, 1], u[:, 0]), phi)

    # build the displaced points near (0, 0, 1) and rotate near theta and phi
    v = xp.stack(
        [
            xp.sin(r) * xp.cos(math.pi - x),
            xp.sin(r) * xp.sin(math.pi - x),
            xp.cos(r),
        ],
        axis=1,
    )
    v = rot @ xp.reshape(v, (n, 3, 1))
    v = xp.reshape(v, (n, 3))

    # compute displaced theta and phi
    theta_d = xp.atan2(xp.hypot(v[:, 0], v[:, 1]), v[:, 2])
    phi_d = xp.atan2(v[:, 1], v[:, 0])

    # compute longitude and latitude
    from_lon = uxpx.degrees(phi)
    from_lat = 90.0 - uxpx.degrees(theta)
    to_lon = uxpx.degrees(phi_d)
    to_lat = 90.0 - uxpx.degrees(theta_d)

    # compute displacement and compare to input
    alpha_in = r * xp.exp(1j * x)
    alpha_out = glass.displacement(from_lon, from_lat, to_lon, to_lat)
    compare.assert_allclose(alpha_out, alpha_in, atol=0.0, rtol=1e-10)
