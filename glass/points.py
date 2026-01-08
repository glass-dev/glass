"""
Random points
=============

.. currentmodule:: glass

The following functions provide functionality for simulating point
processes on the sphere and sampling random positions.

Sampling
--------

.. autofunction:: positions_from_delta
.. autofunction:: uniform_positions
.. autofunction:: position_weights


Bias
----

.. autofunction:: effective_bias


Bias models
-----------

.. autofunction:: linear_bias
.. autofunction:: loglinear_bias


Displacing points
-----------------

.. autofunction:: displace
.. autofunction:: displacement

"""  # noqa: D400

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import healpix
import numpy as np

import array_api_compat

import glass._array_api_utils as _utils
import glass.arraytools
import glass.shells

if TYPE_CHECKING:
    from collections.abc import Callable, Generator
    from types import ModuleType

    from glass._types import (
        ComplexArray,
        FloatArray,
        IntArray,
        UnifiedGenerator,
    )


ARCMIN2_SPHERE = 60**6 // 100 / math.pi


def effective_bias(
    z: FloatArray,
    bz: FloatArray,
    w: glass.shells.RadialWindow,
) -> float | FloatArray:
    r"""
    Effective bias parameter from a redshift-dependent bias function.

    This function takes a redshift-dependent bias function :math:`b(z)`
    and computes an effective bias parameter :math:`\bar{b}` for a
    given window function :math:`w(z)`.

    Parameters
    ----------
    z
        Redshifts and values of the bias function :math:`b(z)`.
    bz
        Redshifts and values of the bias function :math:`b(z)`.
    w
        The radial window function :math:`w(z)`.

    Returns
    -------
        The effective bias parameter for the window.

    Notes
    -----
    The effective bias parameter :math:`\bar{b}` is computed using the
    window function :math:`w(z)` as the weighted average

    .. math::

        \bar{b} = \frac{\int b(z) \, w(z) \, dz}{\int w(z) \, dz}
        \;.

    """
    xp = array_api_compat.array_namespace(z, bz, w.za, w.wa, use_compat=False)
    uxpx = _utils.XPAdditions(xp)

    norm = uxpx.trapezoid(w.wa, w.za)
    return glass.arraytools.trapezoid_product((z, bz), (w.za, w.wa)) / norm


def linear_bias(
    delta: FloatArray,
    b: float | FloatArray,
) -> FloatArray:
    r"""
    Linear bias model :math:`\delta_g = b \, \delta`.

    Parameters
    ----------
    delta
        The input density contrast.
    b
        The bias parameter.

    Returns
    -------
        The density contrast after biasing.

    """
    return b * delta


def loglinear_bias(
    delta: FloatArray,
    b: float | FloatArray,
) -> FloatArray:
    r"""
    Log-linear bias model :math:`\ln(1 + \delta_g) = b \ln(1 + \delta)`.

    Parameters
    ----------
    delta
        The input density contrast.
    b
        The bias parameter.

    Returns
    -------
        The density contrast after biasing.

    """
    xp = array_api_compat.array_namespace(delta, b, use_compat=False)

    delta_g = xp.log1p(delta)
    delta_g *= b
    return xp.expm1(delta_g)


def _broadcast_inputs(
    bias: float | FloatArray | None,
    delta: FloatArray,
    ngal: float | FloatArray,
    vis: FloatArray | None,
) -> tuple[
    float | FloatArray | None,
    FloatArray,
    tuple[int, ...],
    float | FloatArray,
    FloatArray | None,
]:
    """
    Broadcast inputs to common shape of extra dimensions.

    Figure out how many "populations" of objects we are
    dealing with, by broadcasting all inputs together.

    Parameters
    ----------
    bias
        Bias parameter, is passed as an argument to the bias model.
    delta
        Map of the input density contrast. This is fed into the bias
        model to produce the density contrast for sampling.
    ngal
        Number density, expected number of points per arcmin2.
    vis
        Visibility map for the observed points. This is multiplied with
        the full sky number count map, and must hence be of compatible shape.

    Returns
    -------
        The broadcasted inputs.
    """
    inputs: list[tuple[float | FloatArray, int]] = [(ngal, 0), (delta, 1)]
    if bias is not None:
        inputs.append((bias, 0))
    if vis is not None:
        inputs.append((vis, 1))
    dims, *rest = glass.arraytools.broadcast_leading_axes(*inputs)
    ngal, delta, *rest = rest
    if bias is not None:
        bias, *rest = rest
    if vis is not None:
        vis, *rest = rest

    return bias, delta, dims, ngal, vis


def _compute_density_contrast(
    bias: float | FloatArray | None,
    bias_model: Callable[..., Any],
    delta: FloatArray,
    k: tuple[int, ...],
) -> FloatArray:
    """
    Compute density contrast from bias model, or copy.

    Applies the bias model to ``delta``.

    Parameters
    ----------
    bias
        Bias parameter, is passed as an argument to the bias model.
    bias_model
        The bias model to apply. For examples, :func:`glass.linear_bias`
        or :func:`glass.loglinear_bias`.
    delta
        Map of the input density contrast. This is fed into the bias
        model to produce the density contrast for sampling.
    k
        _description_

    Returns
    -------
        The density contrast after biasing.
    """
    return np.copy(delta[k]) if bias is None else bias_model(delta[k], bias[k])  # type: ignore[index]


def _compute_expected_count(
    k: tuple[int, ...],
    n: FloatArray,
    ngal: float | FloatArray,
    *,
    remove_monopole: bool,
) -> FloatArray:
    """
    Computes the expected number of objects per pixel.

    Parameters
    ----------
    k
        _description_
    n
        _description_
    ngal
        Number density, expected number of points per arcmin2.
    remove_monopole
        If true, the monopole of the density contrast
        after biasing is fixed to zero.

    Returns
    -------
        Expected number of objects per pixel.
    """
    # remove monopole if asked to
    if remove_monopole:
        n -= np.mean(n, keepdims=True)

    # turn into number count, modifying the array in place
    n += 1
    n *= ARCMIN2_SPHERE / n.size * ngal[k]  # type: ignore[index]
    return n


def _apply_visibility(
    k: tuple[int, ...],
    n: FloatArray,
    vis: FloatArray | None,
) -> FloatArray:
    """
    Apply visibility if given.

    Parameters
    ----------
    k
        _description_
    n
        _description_
    vis
        Visibility map for the observed points. This is multiplied with
        the full sky number count map, and must hence be of compatible shape.

    Returns
    -------
        The visibility-applied number count map.
    """
    if vis is not None:
        n *= vis[k]
    return n


def _sample_number_galaxies(
    n: FloatArray,
    rng: np.random.Generator,
) -> IntArray:
    """
    Sample the actual number of galaxies in each
    pixel from the Poisson distribution.

    Parameters
    ----------
    n
        _description_
    rng
        Random number generator. If not given, a default RNG is used.

    Returns
    -------
        The sampled number of galaxies per pixel.
    """
    # clip number density at zero
    np.clip(n, 0, None, out=n)  # type: ignore[arg-type,type-var]

    # sample actual number in each pixel
    return rng.poisson(n)


def _sample_galaxies_per_pixel(
    batch: int,
    dims: tuple[int, ...],
    k: tuple[int, ...],
    n: FloatArray,
    rng: np.random.Generator,
) -> Generator[
    tuple[
        FloatArray,
        FloatArray,
        int | IntArray,
    ]
]:
    """
    Sample the individual galaxies in each pixel,
    randomly distributed over sub pixels, in batches.

    Parameters
    ----------
    batch
        Maximum number of positions to yield in one batch.
    dims
        _description_
    k
        _description_
    n
        _description_
    rng
        Random number generator. If not given, a default RNG is used.

    Yields
    ------
    lon
        Columns of longitudes for the sampled points.
    lat
        Columns of latitudes for the sampled points.
    count
        The number of sampled points  If multiple populations are sampled, an
        array of counts in the shape of the extra dimensions is returned.
    """
    # total number of points
    count = n.sum()  # type: ignore[union-attr]
    # don't go through pixels if there are no points
    if count == 0:
        return

    # for converting randomly sampled positions to HEALPix indices
    npix = n.shape[-1]
    nside = healpix.npix2nside(npix)

    # create a mask to report the count in the right axis
    cmask: int | IntArray
    if dims:
        cmask = np.zeros(dims, dtype=int)
        cmask[k] = 1
    else:
        cmask = 1

    # sample the map in batches
    step = 1_000
    start, stop, size = 0, 0, 0
    while count:
        # tally this group of pixels
        q = np.cumulative_sum(n[stop : stop + step])
        # does this group of pixels fill the batch?
        if size + q[-1] < min(batch, count):
            # no, we need the next group of pixels to fill the batch
            stop += step
            size += q[-1]
        else:
            # how many pixels from this group do we need?
            stop += int(np.searchsorted(q, batch - size, side="right"))
            # if the first pixel alone is too much, use it anyway
            if stop == start:
                stop += 1
            # sample this batch of pixels
            ipix = np.repeat(np.arange(start, stop), n[start:stop])  # type: ignore[arg-type]
            lon, lat = healpix.randang(nside, ipix, lonlat=True, rng=rng)
            # next batch
            start, size = stop, 0
            # keep track of remaining number of points
            count -= ipix.size
            # yield the batch
            yield lon, lat, ipix.size * cmask

    # make sure that the correct number of pixels was sampled
    assert np.sum(n[stop:]) == 0  # noqa: S101


def positions_from_delta(  # noqa: PLR0913
    ngal: float | FloatArray,
    delta: FloatArray,
    bias: float | FloatArray | None = None,
    vis: FloatArray | None = None,
    *,
    bias_model: Callable[..., Any] = linear_bias,
    remove_monopole: bool = False,
    batch: int = 1_000_000,
    rng: np.random.Generator | None = None,
) -> Generator[
    tuple[
        FloatArray,
        FloatArray,
        int | IntArray,
    ]
]:
    """
    Generate positions tracing a density contrast.

    The map of expected number counts is constructed from the number
    density, density contrast, an optional bias model, and an optional
    visibility map.

    If ``remove_monopole`` is set, the monopole of the computed density
    contrast is removed. Over the full sky, the mean number density of
    the map will then match the given number density exactly. This,
    however, means that an effectively different bias model is being
    used, unless the monopole is already zero in the first place.

    The function supports multi-dimensional input for the ``ngal``,
    ``delta``, ``bias``, and ``vis`` parameters. Extra dimensions are
    broadcast to a common shape, and treated as separate populations of
    points. These are then sampled independently, and the results
    concatenated into a flat list of longitudes and latitudes. The
    number of points per population is returned in ``count`` as an array
    in the shape of the extra dimensions.

    Parameters
    ----------
    ngal
        Number density, expected number of points per arcmin2.
    delta
        Map of the input density contrast. This is fed into the bias
        model to produce the density contrast for sampling.
    bias
        Bias parameter, is passed as an argument to the bias model.
    vis
        Visibility map for the observed points. This is multiplied with
        the full sky number count map, and must hence be of compatible shape.
    bias_model
        The bias model to apply. For examples, :func:`glass.linear_bias`
        or :func:`glass.loglinear_bias`.
    remove_monopole
        If true, the monopole of the density contrast
        after biasing is fixed to zero.
    batch
        Maximum number of positions to yield in one batch.
    rng
        Random number generator. If not given, a default RNG is used.

    Yields
    ------
    lon
        Columns of longitudes for the sampled points.
    lat
        Columns of latitudes for the sampled points.
    count
        The number of sampled points  If multiple populations are sampled, an
        array of counts in the shape of the extra dimensions is returned.

    Raises
    ------
    TypeError
        If the bias model is not a string or callable.

    """
    # get default RNG if not given
    if rng is None:
        rng = np.random.default_rng(42)

    # ensure bias_model is a function
    if not callable(bias_model):
        raise TypeError("bias_model must be callable")

    bias, delta, dims, ngal, vis = _broadcast_inputs(bias, delta, ngal, vis)

    # iterate the leading dimensions
    for k in np.ndindex(dims):
        n = _compute_density_contrast(bias, bias_model, delta, k)

        n = _compute_expected_count(k, n, ngal, remove_monopole=remove_monopole)

        n = _apply_visibility(k, n, vis)

        n = _sample_number_galaxies(n, rng)

        yield from _sample_galaxies_per_pixel(batch, dims, k, n, rng)


def uniform_positions(
    ngal: float | IntArray | FloatArray,
    *,
    rng: UnifiedGenerator | None = None,
    xp: ModuleType | None = None,
) -> Generator[
    tuple[
        FloatArray,
        FloatArray,
        int | IntArray,
    ]
]:
    """
    Generate positions uniformly over the sphere.

    The function supports array input for the ``ngal`` parameter.

    Parameters
    ----------
    ngal
        Number density, expected number of positions per arcmin2.
    rng
        Random number generator. If not given, a default RNG is used.

    Yields
    ------
    lon
        Columns of longitudes for the sampled points.
    lat
        Columns of latitudes for the sampled points.
    count
        The number of sampled points. For array inputs, an array of
        counts with the same shape is returned.

    """
    if xp is None:
        xp = array_api_compat.array_namespace(ngal, use_compat=False)
    uxpx = _utils.XPAdditions(xp)

    # get default RNG if not given
    if rng is None:
        rng = _utils.rng_dispatcher(xp=xp)

    ngal = xp.asarray(ngal)

    # sample number of galaxies
    ngal_sphere = xp.asarray(rng.poisson(xp.multiply(ARCMIN2_SPHERE, ngal)))

    # extra dimensions of the output
    dims = ngal_sphere.shape

    # sample each set of points
    for k in uxpx.ndindex(dims):
        size = (ngal_sphere[k],)
        # sample uniformly over the sphere
        lon = rng.uniform(-180, 180, size=size)
        lat = uxpx.degrees(xp.asin(rng.uniform(-1, 1, size=size)))

        # report count
        count: int | IntArray
        if dims:
            count = xp.zeros(dims, dtype=xp.int64)
            count[k] = ngal_sphere[k]  # type: ignore[index]
        else:
            count = int(ngal_sphere[k])

        yield lon, lat, count


def position_weights(
    densities: FloatArray,
    bias: FloatArray | float | None = None,
) -> FloatArray:
    r"""
    Compute relative weights for angular clustering.

    Takes an array *densities* of densities in arbitrary units and
    returns the relative weight of each shell. If *bias* is given, a
    linear bias is applied to each shell.

    This is the equivalent of computing the product of normalised
    redshift distribution and bias factor :math:`n(z) \, b(z)` for the
    discretised shells.

    Parameters
    ----------
    densities
        Density of points in each shell. The first axis must broadcast
        against the number of shells, and is normalised internally.
    bias
        Value or values of the linear bias parameter for each shell.

    Returns
    -------
        The relative weight of each shell for angular clustering.

    """
    xp = array_api_compat.array_namespace(densities, bias, use_compat=False)

    bias = bias if bias is None or not isinstance(bias, float) else xp.asarray(bias)

    # bring densities and bias into the same shape
    if bias is not None:
        densities, bias = glass.arraytools.broadcast_first(densities, bias)
    # normalise densities after shape has been fixed
    densities = densities / xp.sum(densities, axis=0)
    # apply bias after normalisation
    if bias is not None:
        densities = densities * bias
    # densities now contains the relative contribution with bias applied
    return densities


def displace(
    lon: FloatArray,
    lat: FloatArray,
    alpha: ComplexArray | FloatArray,
) -> tuple[FloatArray, FloatArray]:
    r"""
    Displace positions on the sphere.

    Takes an array of :term:`displacement` values and applies them to
    the given positions.

    Parameters
    ----------
    lon
        Longitudes to be displaced.
    lat
        Latitudes to be displaced.
    alpha
        Displacement values. Must be complex-valued or have a leading
        axis of size 2 for the real and imaginary component.

    Returns
    -------
        The longitudes and latitudes after displacement.

    Notes
    -----
    Displacements on the sphere are :term:`defined <displacement>` as
    follows:  The complex displacement :math:`\alpha` transports a point
    on the sphere an angular distance :math:`|\alpha|` along the
    geodesic with bearing :math:`\arg\alpha` in the original point.

    In the language of differential geometry, this function is the
    exponential map.

    """
    xp = array_api_compat.get_namespace(lon, lat, alpha, use_compat=False)

    alpha = xp.asarray(alpha)
    if xp.isdtype(alpha.dtype, "complex floating"):
        alpha1, alpha2 = xp.real(alpha), xp.imag(alpha)
    else:
        alpha1, alpha2 = alpha

    # we know great-circle navigation:
    # θ' = arctan2(√[(cosθ sin|α| - sinθ cos|α| cosγ)² + (sinθ sinγ)²],
    #              cosθ cos|α| + sinθ sin|α| cosγ)
    # δ = arctan2(sin|α| sinγ, sinθ cos|α| - cosθ sin|α| cosγ)

    t = xp.asarray(lat) / 180 * math.pi
    ct, st = xp.sin(t), xp.cos(t)  # sin and cos flipped: lat not co-lat

    a = xp.hypot(alpha1, alpha2)  # abs(alpha)
    g = xp.atan2(alpha2, alpha1)  # arg(alpha)
    ca, sa = xp.cos(a), xp.sin(a)
    cg, sg = xp.cos(g), xp.sin(g)

    # flipped atan2 arguments for lat instead of co-lat
    tp = xp.atan2(ct * ca + st * sa * cg, xp.hypot(ct * sa - st * ca * cg, st * sg))

    d = xp.atan2(sa * sg, st * ca - ct * sa * cg)

    return lon - d / math.pi * 180, tp / math.pi * 180


def displacement(
    from_lon: FloatArray,
    from_lat: FloatArray,
    to_lon: FloatArray,
    to_lat: FloatArray,
) -> ComplexArray:
    """
    Compute the displacement between two sets of positions.

    Compute the complex :term:`displacement` that transforms points with
    longitude *from_lon* and latitude *from_lat* into points with
    longitude *to_lon* and latitude *to_lat* (all in degrees).

    Parameters
    ----------
    from_lon, from_lat
        Points before displacement.
    to_lon, to_lat
        Points after displacement.

    Returns
    -------
        Array of complex displacement.

    See Also
    --------
    displace : Apply displacement to a set of points.

    """
    xp = array_api_compat.get_namespace(
        from_lon,
        from_lat,
        to_lon,
        to_lat,
        use_compat=False,
    )

    a = (90.0 - to_lat) / 180 * math.pi
    b = (90.0 - from_lat) / 180 * math.pi
    g = (from_lon - to_lon) / 180 * math.pi

    sa, ca = xp.sin(a), xp.cos(a)
    sb, cb = xp.sin(b), xp.cos(b)
    sg, cg = xp.sin(g), xp.cos(g)

    r = xp.atan2(xp.hypot(sa * cb - ca * sb * cg, sb * sg), ca * cb + sa * sb * cg)
    x = sb * ca - cb * sa * cg
    y = sa * sg
    z = xp.hypot(x, y)
    return r * (x / z + 1j * y / z)
