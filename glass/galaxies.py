# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''
Galaxies (:mod:`glass.galaxies`)
================================

.. currentmodule:: glass.galaxies

The :mod:`glass.galaxies` module provides functionality for simulating galaxies
as typically observed in a cosmological galaxy survey.

Functions
---------

.. autofunction:: redshifts_from_nz
.. autofunction:: galaxy_shear
.. autofunction:: gaussian_phz

'''

import numpy as np
import healpix

from typing import Optional, Tuple

from .math import cumtrapz


def redshifts_from_nz(size: int, z: np.ndarray, nz: np.ndarray, *,
                      rng: Optional[np.random.Generator] = None
                      ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    '''Generate galaxy redshifts from a source distribution.

    Parameters
    ----------
    size : int
        Number of redshifts to sample.
    z, nz : array_like
        Source distribution.  Leading axes are treated as different
        galaxy populations.
    rng : :class:`~numpy.random.Generator`, optional
        Random number generator.  If not given, a default RNG will be
        used.

    Returns
    -------
    z : array_like
        Redshifts sampled from the given source distribution.
    pop : array_like or None
        Index of the galaxy population from the leading axes of ``nz``;
        or ``None`` if there are no galaxy populations.

    '''

    # get default RNG if not given
    if rng is None:
        rng = np.random.default_rng()

    # get galaxy populations
    if np.ndim(nz) > 1:
        pop = list(np.ndindex(np.shape(nz)[:-1]))
    else:
        pop = None

    # compute the as-yet unnormalised CDF of each galaxy population
    cdf = cumtrapz(nz, z)

    # compute probability to be in each galaxy population
    p = cdf[..., -1]/cdf[..., -1].sum(axis=None, keepdims=True)

    # now normalise the CDFs
    cdf /= cdf[..., -1:]

    # sample redshifts and populations
    if pop is not None:
        x = rng.choice(len(pop), p=p, size=size)
        gal_z = rng.uniform(0, 1, size=size)
        for i, j in enumerate(pop):
            s = (x == i)
            gal_z[s] = np.interp(gal_z[s], cdf[j], z)
        gal_pop = np.take(pop, x)
    else:
        gal_z = np.interp(rng.uniform(0, 1, size=size), cdf, z)
        gal_pop = None

    return gal_z, gal_pop


def galaxy_shear(lon: np.ndarray, lat: np.ndarray, eps: np.ndarray,
                 kappa: np.ndarray, gamma1: np.ndarray, gamma2: np.ndarray, *,
                 reduced_shear: bool = True) -> np.ndarray:
    '''Observed galaxy shears from weak lensing.

    Takes lensing maps for convergence and shear and produces a lensed
    ellipticity (shear) for each intrinsic galaxy ellipticity.

    Parameters
    ----------
    lon, lat : array_like
        Arrays for galaxy longitudes and latitudes.
    eps : array_like
        Array of galaxy :term:`ellipticity`.
    kappa, gamma1, gamma2 : array_like
        HEALPix maps for convergence and two components of shear.
    reduced_shear : bool, optional
        If ``False``, galaxy shears are not reduced by the convergence.
        Default is ``True``.

    Returns
    -------
    she : array_like
        Array of complex-valued observed galaxy shears (lensed ellipticities).

    '''

    nside = healpix.npix2nside(np.broadcast(kappa, gamma1, gamma2).shape[-1])

    size = np.broadcast(lon, lat, eps).size

    # output arrays
    k = np.empty(size)
    g = np.empty(size, dtype=complex)

    # get the lensing maps at galaxy position
    for i in range(0, size, 10000):
        s = slice(i, i+10000)
        ipix = healpix.ang2pix(nside, lon[s], lat[s], lonlat=True)
        k[s] = kappa[ipix]
        g.real[s] = gamma1[ipix]
        g.imag[s] = gamma2[ipix]

    if reduced_shear:
        # compute reduced shear in place
        g /= 1 - k

        # compute lensed ellipticities
        g = (eps + g)/(1 + g.conj()*eps)
    else:
        # simple sum of shears
        g += eps

    return g


def gaussian_phz(z: np.ndarray, sigma_0: float,
                 rng: Optional[np.random.Generator] = None) -> np.ndarray:
    r'''Photometric redshifts assuming a Gaussian error.

    A simple toy model of photometric redshift errors that assumes a Gaussian
    error with redshift-dependent standard deviation :math:`\sigma(z) = (1 + z)
    \sigma_0` [1]_.

    Parameters
    ----------
    z : array_like
        True redshifts.
    sigma_0 : float
        Redshift error in the tomographic binning at zero redshift.
    rng : :class:`~numpy.random.Generator`, optional
        Random number generator.  If not given, a default RNG will be used.

    Returns
    -------
    phz : array_like
        Photometric redshifts assuming Gaussian errors, of the same shape as
        ``z``.

    See Also
    --------
    glass.observations.tomo_nz_gausserr :
        Create tomographic redshift distributions assuming the same model.

    References
    ----------
    .. [1] Amara A., Réfrégier A., 2007, MNRAS, 381, 1018.
           doi:10.1111/j.1365-2966.2007.12271.x

    Examples
    --------
    See the :doc:`examples:basic/photoz` example.

    '''

    # get default RNG if not given
    if rng is None:
        rng = np.random.default_rng()

    size = np.shape(z)
    z = np.reshape(z, (-1,))

    zphot = rng.normal(z, (1 + z)*sigma_0)

    trunc = np.where(zphot < 0)[0]
    while trunc.size:
        zphot[trunc] = rng.normal(z[trunc], (1 + z[trunc])*sigma_0)
        trunc = trunc[zphot[trunc] < 0]

    return zphot.reshape(size)
