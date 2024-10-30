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

"""  # noqa: D205, D400

import healpix
import numpy as np

from glass.core.array import broadcast_first, broadcast_leading_axes, trapz_product

ARCMIN2_SPHERE = 60**6 // 100 / np.pi


def effective_bias(z, bz, w):  # type: ignore[no-untyped-def]
    r"""
    Effective bias parameter from a redshift-dependent bias function.

    This function takes a redshift-dependent bias function :math:`b(z)`
    and computes an effective bias parameter :math:`\bar{b}` for a
    given window function :math:`w(z)`.

    Returns the effective bias parameter for the window.

    Parameters
    ----------
    z:
        Redshifts and values of the bias function :math:`b(z)`.
    bz:
        Redshifts and values of the bias function :math:`b(z)`.
    w:
        The radial window function :math:`w(z)`.

    Notes
    -----
    The effective bias parameter :math:`\bar{b}` is computed using the
    window function :math:`w(z)` as the weighted average

    .. math::

        \bar{b} = \frac{\int b(z) \, w(z) \, dz}{\int w(z) \, dz}
        \;.

    """
    norm = np.trapz(w.wa, w.za)  # type: ignore[attr-defined]
    return trapz_product((z, bz), (w.za, w.wa)) / norm  # type: ignore[no-untyped-call]


def linear_bias(delta, b):  # type: ignore[no-untyped-def]
    r"""Linear bias model :math:`\delta_g = b \, \delta`."""
    return b * delta


def loglinear_bias(delta, b):  # type: ignore[no-untyped-def]
    r"""log-linear bias model :math:`\ln(1 + \delta_g) = b \ln(1 + \delta)`."""
    delta_g = np.log1p(delta)
    delta_g *= b
    np.expm1(delta_g, out=delta_g)
    return delta_g


def positions_from_delta(  # type: ignore[no-untyped-def] # noqa: PLR0912, PLR0913, PLR0915
    ngal,
    delta,
    bias=None,
    vis=None,
    *,
    bias_model="linear",
    remove_monopole=False,
    batch=1_000_000,
    rng=None,
):
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
    ngal:
        Number density, expected number of points per arcmin2.
    delta:
        Map of the input density contrast. This is fed into the bias
        model to produce the density contrast for sampling.
    bias:
        Bias parameter, is passed as an argument to the bias model.
    vis:
        Visibility map for the observed points. This is multiplied with
        the full sky number count map, and must hence be of compatible shape.
    bias_model:
        The bias model to apply. If a string, refers to a function in
        the :mod:`~glass.points` module, e.g. ``'linear'`` for
        :func:`linear_bias()` or ``'loglinear'`` for :func:`loglinear_bias`.
    remove_monopole:
        If true, the monopole of the density contrast
        after biasing is fixed to zero.
    batch:
        Maximum number of positions to yield in one batch.
    rng:
        Random number generator. If not given, a default RNG is used.

    Yields
    ------
    lon:
        Columns of longitudes for the sampled points.
    lat:
        Columns of latitudes for the sampled points.
    count:
        The number of sampled points  If multiple populations are sampled, an
        array of counts in the shape of the extra dimensions is returned.

    """
    # get default RNG if not given
    if rng is None:
        rng = np.random.default_rng()

    # get the bias model
    if isinstance(bias_model, str):
        bias_model = globals()[f"{bias_model}_bias"]
    elif not callable(bias_model):
        msg = "bias_model must be string or callable"
        raise TypeError(msg)

    # broadcast inputs to common shape of extra dimensions
    inputs = [(ngal, 0), (delta, 1)]
    if bias is not None:
        inputs += [(bias, 0)]
    if vis is not None:
        inputs += [(vis, 1)]
    dims, ngal, delta, *rest = broadcast_leading_axes(*inputs)  # type: ignore[no-untyped-call]
    if bias is not None:
        bias, *rest = rest
    if vis is not None:
        vis, *rest = rest

    # iterate the leading dimensions
    for k in np.ndindex(dims):
        # compute density contrast from bias model, or copy
        n = np.copy(delta[k]) if bias is None else bias_model(delta[k], bias[k])

        # remove monopole if asked to
        if remove_monopole:
            n -= np.mean(n, keepdims=True)

        # turn into number count, modifying the array in place
        n += 1
        n *= ARCMIN2_SPHERE / n.size * ngal[k]

        # apply visibility if given
        if vis is not None:
            n *= vis[k]

        # clip number density at zero
        np.clip(n, 0, None, out=n)

        # sample actual number in each pixel
        n = rng.poisson(n)

        # total number of points
        count = n.sum()
        # don't go through pixels if there are no points
        if count == 0:
            continue

        # for converting randomly sampled positions to HEALPix indices
        npix = n.shape[-1]
        nside = healpix.npix2nside(npix)

        # create a mask to report the count in the right axis
        if dims:
            cmask = np.zeros(dims, dtype=int)
            cmask[k] = 1
        else:
            cmask = 1  # type: ignore[assignment]

        # sample the map in batches
        step = 1000
        start, stop, size = 0, 0, 0
        while count:
            # tally this group of pixels
            q = np.cumsum(n[stop : stop + step])
            # does this group of pixels fill the batch?
            if size + q[-1] < min(batch, count):
                # no, we need the next group of pixels to fill the batch
                stop += step
                size += q[-1]
            else:
                # how many pixels from this group do we need?
                stop += np.searchsorted(q, batch - size, side="right")
                # if the first pixel alone is too much, use it anyway
                if stop == start:
                    stop += 1
                # sample this batch of pixels
                ipix = np.repeat(np.arange(start, stop), n[start:stop])
                lon, lat = healpix.randang(nside, ipix, lonlat=True, rng=rng)
                # next batch
                start, size = stop, 0
                # keep track of remaining number of points
                count -= ipix.size
                # yield the batch
                yield lon, lat, ipix.size * cmask

        # make sure that the correct number of pixels was sampled
        assert np.sum(n[stop:]) == 0  # noqa: S101


def uniform_positions(ngal, *, rng=None):  # type: ignore[no-untyped-def]
    """
    Generate positions uniformly over the sphere.

    The function supports array input for the ``ngal`` parameter.

    Parameters
    ----------
    ngal:
        Number density, expected number of positions per arcmin2.
    rng:
        Random number generator. If not given, a default RNG will be used.

    Yields
    ------
    lon:
        Columns of longitudes for the sampled points.
    lat:
        Columns of latitudes for the sampled points.
    count:
        The number of sampled points. For array inputs, an array of
        counts with the same shape is returned.

    """
    # get default RNG if not given
    if rng is None:
        rng = np.random.default_rng()

    # sample number of galaxies
    ngal = rng.poisson(np.multiply(ARCMIN2_SPHERE, ngal))

    # extra dimensions of the output
    dims = np.shape(ngal)

    # make sure ntot is an array even if scalar
    ngal = np.broadcast_to(ngal, dims)

    # sample each set of points
    for k in np.ndindex(dims):
        # sample uniformly over the sphere
        lon = rng.uniform(-180, 180, size=ngal[k])
        lat = np.rad2deg(np.arcsin(rng.uniform(-1, 1, size=ngal[k])))

        # report count
        if dims:
            count = np.zeros(dims, dtype=int)
            count[k] = ngal[k]
        else:
            count = int(ngal[k])  # type: ignore[assignment]

        yield lon, lat, count


def position_weights(densities, bias=None):  # type: ignore[no-untyped-def]
    r"""
    Compute relative weights for angular clustering.

    Takes an array *densities* of densities in arbitrary units and
    returns the relative weight of each shell. If *bias* is given, a
    linear bias is applied to each shell.

    This is the equivalent of computing the product of normalised
    redshift distribution and bias factor :math:`n(z) \, b(z)` for the
    discretised shells.

    Returns the relative weight of each shell for angular clustering.

    Parameters
    ----------
    densities:
        Density of points in each shell. The first axis must broadcast
        against the number of shells, and is normalised internally.
    bias:
        Value or values of the linear bias parameter for each shell.

    """
    # bring densities and bias into the same shape
    if bias is not None:
        densities, bias = broadcast_first(densities, bias)  # type: ignore[no-untyped-call]
    # normalise densities after shape has been fixed
    densities = densities / np.sum(densities, axis=0)
    # apply bias after normalisation
    if bias is not None:
        densities = densities * bias
    # densities now contains the relative contribution with bias applied
    return densities
