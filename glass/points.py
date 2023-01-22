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

.. autosummary::
   :template: generator.rst
   :toctree: generated/
   :nosignatures:

   positions_from_delta
   uniform_positions


Bias
----

.. autosummary::
   :template: generator.rst
   :toctree: generated/
   :nosignatures:

   effective_bias
   linear_bias
   loglinear_bias

'''

import numpy as np
import healpix

from .math import ARCMIN2_SPHERE, restrict_interval, trapz_product


def effective_bias(z, b, shells, weights):
    '''Effective bias parameter from a redshift-dependent bias function.

    This function takes a redshift-dependent bias function :math:`b(z)` and
    computes an effective bias parameter :math:`\\bar{b}_i` for each shell
    :math:`i` using the matter weight function.

    Parameters
    ----------
    z, b : array_like
        Redshifts and values of bias function :math:`b(z)`.
    shells : array_like
        Redshifts of the shell boundaries.
    weights : :class:`glass.matter.MatterWeights`
        The matter weight function for the shells.

    Returns
    -------
    beff : array_like
        Effective bias parameter for each shell.

    Notes
    -----
    The effective bias parameter :math:`\\bar{b}_i` in shell :math:`i` is
    computed using the matter weight function :math:`W` as the weighted
    average

    .. math::

        \\bar{b}_i = \\frac{\\int_{z_{i-1}}^{z_i} b(z) \\, W(z) \\, dz}
                           {\\int_{z_{i-1}}^{z_i} W(z) \\, dz}  \\;.

    '''
    beff = np.empty(len(shells)-1)
    for i, (zmin, zmax) in enumerate(zip(shells, shells[1:])):
        w_, z_ = restrict_interval(weights.w, weights.z, zmin, zmax)
        beff[i] = trapz_product((z, b), (z_, w_))/np.trapz(w_, z_)
    return beff


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

    The map of expected number counts is constructed from the number density,
    density contrast, an optional bias model, and an optional visibility map.

    If ``remove_monopole`` is set, the monopole of the computed density contrast
    is removed.  Over the full sky, the mean number density of the map will then
    match the given number density exactly.  This, however, means that an
    effectively different bias model is being used, unless the monopole is
    already zero in the first place.

    Parameters
    ----------
    ngal : float
        Number density, expected number of points per arcmin2.
    delta : array_like
        Map of the input density contrast.  This is fed into the bias model to
        produce the density contrast for sampling.
    bias : float, optional
        Bias parameter, is passed as an argument to the bias model.
    vis : array_like, optional
        Visibility map for the observed points.  This is multiplied with the
        full sky number count map, and must hence be of compatible shape.
    bias_model : str or callable, optional
        The bias model to apply.  If a string, refers to a function in the
        points module, e.g. ``'linear'`` for ``glass.points.linear_bias`` or
        ``'loglinear'`` for ``glass.points.loglinear_bias``.
    remove_monopole : bool, optional
        If true, the monopole of the density contrast after biasing is fixed to
        zero.
    rng : :class:`~numpy.random.Generator`, optional
        Random number generator.  If not given, a default RNG will be used.

    Returns
    -------
    lon, lat : array_like
        Columns of longitudes and latitudes for the sampled points.

    '''

    # get default RNG if not given
    if rng is None:
        rng = np.random.default_rng()

    # get the bias model
    if isinstance(bias_model, str):
        bias_model = globals()[f'{bias_model}_bias']
    elif not callable(bias_model):
        raise ValueError('bias_model must be string or callable')

    # compute density contrast from bias model, or copy
    if bias is None:
        n = np.copy(delta)
    else:
        n = bias_model(delta, bias)

    # remove monopole if asked to
    if remove_monopole:
        n -= np.mean(n, keepdims=True)

    # turn into number count, modifying the array in place
    n += 1
    n *= ARCMIN2_SPHERE/n.size*ngal

    # apply visibility if given
    if vis is not None:
        n *= vis

    # clip number density at zero
    np.clip(n, 0, None, out=n)

    # sample actual number in each pixel
    n = rng.poisson(n)

    # total number of sampled points
    ntot = n.sum()

    # for converting randomly sampled positions to HEALPix indices
    npix = n.shape[-1]
    nside = healpix.npix2nside(npix)

    # these will hold the results
    lon = np.empty(ntot)
    lat = np.empty(ntot)

    # sample batches of 10000 pixels
    batch = 10_000
    ncur = 0
    for i in range(0, npix, batch):
        k = n[i:i+batch]
        bpix = np.repeat(np.arange(i, i+k.size), k)
        blon, blat = healpix.randang(nside, bpix, lonlat=True, rng=rng)
        lon[ncur:ncur+blon.size] = blon
        lat[ncur:ncur+blat.size] = blat
        ncur += bpix.size

    assert ncur == ntot, 'internal error in sampling'

    return lon, lat


def uniform_positions(ngal, *, rng=None):
    '''Generate positions uniformly over the sphere.

    Parameters
    ----------
    ngal : float
        Number density, expected number of positions per arcmin2.
    rng : :class:`~numpy.random.Generator`, optional
        Random number generator.  If not given, a default RNG will be used.

    Returns
    -------
    lon, lat : array_like
        Columns of longitudes and latitudes for the sampled points.

    '''

    # get default RNG if not given
    if rng is None:
        rng = np.random.default_rng()

    # sample number of galaxies
    ntot = rng.poisson(ARCMIN2_SPHERE*ngal)

    # sample uniformly over the sphere
    lon = rng.uniform(-180, 180, size=ntot)
    lat = np.rad2deg(np.arcsin(rng.uniform(-1, 1, size=ntot)))

    return lon, lat
