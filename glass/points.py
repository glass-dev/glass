# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''
Random points (:mod:`glass.points`)
===================================

.. currentmodule:: glass.points

The :mod:`glass.points` module provides functionality for simulating point
processes on the sphere and sampling random positions.

Sampling
--------

.. autofunction:: positions_from_delta
.. autofunction:: uniform_positions


Bias
----

.. autofunction:: effective_bias


Bias models
-----------

.. autofunction:: linear_bias
.. autofunction:: loglinear_bias

'''

import numpy as np
import healpix

from .core.array import broadcast_leading_axes, trapz_product
from .core.constants import ARCMIN2_SPHERE


def effective_bias(z, bz, w):
    '''Effective bias parameter from a redshift-dependent bias function.

    This function takes a redshift-dependent bias function :math:`b(z)`
    and computes an effective bias parameter :math:`\\bar{b}` for a
    given window function :math:`w(z)`.

    Parameters
    ----------
    z, bz : array_like
        Redshifts and values of the bias function :math:`b(z)`.
    w : :class:`~glass.shells.RadialWindow`
        The radial window function :math:`w(z)`.

    Returns
    -------
    beff : array_like
        Effective bias parameter for the window.

    Notes
    -----
    The effective bias parameter :math:`\\bar{b}` is computed using the
    window function :math:`w(z)` as the weighted average

    .. math::

        \\bar{b} = \\frac{\\int b(z) \\, w(z) \\, dz}{\\int w(z) \\, dz}
        \\;.

    '''
    norm = np.trapz(w.wa, w.za)
    return trapz_product((z, bz), (w.za, w.wa))/norm


def linear_bias(delta, b):
    '''linear bias model :math:`\\delta_g = b \\, \\delta`'''
    return b*delta


def loglinear_bias(delta, b):
    '''log-linear bias model :math:`\\ln(1 + \\delta_g) = b \\ln(1 + \\delta)`'''
    delta_g = np.log1p(delta)
    delta_g *= b
    np.expm1(delta_g, out=delta_g)
    return delta_g


def positions_from_delta(ngal, delta, bias=None, vis=None, *,
                         bias_model='linear', remove_monopole=False, rng=None):
    '''Generate positions tracing a density contrast.

    The map of expected number counts is constructed from the number
    density, density contrast, an optional bias model, and an optional
    visibility map.

    If ``remove_monopole`` is set, the monopole of the computed density
    contrast is removed.  Over the full sky, the mean number density of
    the map will then match the given number density exactly.  This,
    however, means that an effectively different bias model is being
    used, unless the monopole is already zero in the first place.

    The function supports multi-dimensional input for the ``ngal``,
    ``delta``, ``bias``, and ``vis`` parameters.  Extra dimensions are
    broadcast to a common shape, and treated as separate populations of
    points.  These are then sampled independently, and the results
    concatenated into a flat list of longitudes and latitudes.  The
    number of points per population is returned in ``count`` as an array
    in the shape of the extra dimensions.

    Parameters
    ----------
    ngal : float or array_like
        Number density, expected number of points per arcmin2.
    delta : array_like
        Map of the input density contrast.  This is fed into the bias
        model to produce the density contrast for sampling.
    bias : float or array_like, optional
        Bias parameter, is passed as an argument to the bias model.
    vis : array_like, optional
        Visibility map for the observed points.  This is multiplied with
        the full sky number count map, and must hence be of compatible
        shape.
    bias_model : str or callable, optional
        The bias model to apply.  If a string, refers to a function in
        the :mod:`~glass.points` module, e.g. ``'linear'`` for
        :func:`linear_bias()` or ``'loglinear'`` for
        :func:`loglinear_bias`.
    remove_monopole : bool, optional
        If true, the monopole of the density contrast after biasing is
        fixed to zero.
    rng : :class:`~numpy.random.Generator`, optional
        Random number generator.  If not given, a default RNG is used.

    Returns
    -------
    lon, lat : array_like
        Columns of longitudes and latitudes for the sampled points.
    count : int or array_like
        The number of sampled points.  If multiple populations are
        sampled, an array of counts in the shape of the extra
        dimensions is returned.

    '''

    # get default RNG if not given
    if rng is None:
        rng = np.random.default_rng()

    # get the bias model
    if isinstance(bias_model, str):
        bias_model = globals()[f'{bias_model}_bias']
    elif not callable(bias_model):
        raise ValueError('bias_model must be string or callable')

    # broadcast inputs to common shape of extra dimensions
    inputs = [(ngal, 0), (delta, 1)]
    if bias is not None:
        inputs += [(bias, 0)]
    if vis is not None:
        inputs += [(vis, 1)]
    dims, ngal, delta, *rest = broadcast_leading_axes(*inputs)
    if bias is not None:
        bias, *rest = rest
    if vis is not None:
        vis, *rest = rest

    # the output arrays, concatenated over all dimensions
    ntot = 0
    lon = np.empty(0)
    lat = np.empty(0)

    # keep track of counts for each leading dimensions
    count = np.empty(dims, dtype=int)

    # iterate the leading dimensions
    for k in np.ndindex(dims):

        # compute density contrast from bias model, or copy
        if bias is None:
            n = np.copy(delta[k])
        else:
            n = bias_model(delta[k], bias[k])

        # remove monopole if asked to
        if remove_monopole:
            n -= np.mean(n, keepdims=True)

        # turn into number count, modifying the array in place
        n += 1
        n *= ARCMIN2_SPHERE/n.size*ngal[k]

        # apply visibility if given
        if vis is not None:
            n *= vis[k]

        # clip number density at zero
        np.clip(n, 0, None, out=n)

        # sample actual number in each pixel
        n = rng.poisson(n)

        # for converting randomly sampled positions to HEALPix indices
        npix = n.shape[-1]
        nside = healpix.npix2nside(npix)

        # number of points for this population
        count[k] = n.sum()

        # current and new total number of sampled points
        ncur, ntot = ntot, ntot + count[k]

        # resize the output arrays to hold the new sample
        lon.resize(ntot, refcheck=False)
        lat.resize(ntot, refcheck=False)

        # sample batches of 10000 pixels
        batch = 10_000
        for i in range(0, npix, batch):
            r = n[i:i+batch]
            bpix = np.repeat(np.arange(i, i+r.size), r)
            blon, blat = healpix.randang(nside, bpix, lonlat=True, rng=rng)
            lon[ncur:ncur+blon.size] = blon
            lat[ncur:ncur+blat.size] = blat
            ncur += bpix.size

        assert ncur == ntot, 'internal error in sampling'

    # return a plain scalar of counts if there are no dims
    if not dims:
        count = count.item()

    return lon, lat, count


def uniform_positions(ngal, *, rng=None):
    '''Generate positions uniformly over the sphere.

    The function supports array input for the ``ngal`` parameter.

    Parameters
    ----------
    ngal : float or array_like
        Number density, expected number of positions per arcmin2.
    rng : :class:`~numpy.random.Generator`, optional
        Random number generator.  If not given, a default RNG will be used.

    Returns
    -------
    lon, lat : array_like or list of array_like
        Columns of longitudes and latitudes for the sampled points.
    count : int or list of ints
        The number of sampled points.  For array inputs, an array of
        counts with the same shape is returned.

    '''

    # get default RNG if not given
    if rng is None:
        rng = np.random.default_rng()

    # sample number of galaxies
    count = rng.poisson(np.multiply(ARCMIN2_SPHERE, ngal))

    # extra dimensions of the output
    dims = np.shape(count)

    # make sure ntot is an array even if scalar
    count = np.broadcast_to(count, dims)

    # arrays for results
    ntot = 0
    lon = np.empty(0)
    lat = np.empty(0)

    # sample each set of points
    for k in np.ndindex(dims):

        # resize output arrays
        ncur, ntot = ntot, ntot + count[k]
        lon.resize(ntot, refcheck=False)
        lat.resize(ntot, refcheck=False)

        # sample uniformly over the sphere
        lon[ncur:ntot] = rng.uniform(-180, 180, size=count[k])
        lat[ncur:ntot] = np.rad2deg(np.arcsin(rng.uniform(-1, 1, size=count[k])))

    # return plain scalar if there are no dims
    if not dims:
        count = count.item()

    return lon, lat, count
