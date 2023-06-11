# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''
Observed shapes (:mod:`glass.shapes`)
=====================================

.. currentmodule:: glass.shapes

The :mod:`glass.shapes` module provides functionality for simulating the
observed shapes of objects, such as e.g. galaxies.

Ellipticity
-----------

.. autofunction:: ellipticity_gaussian
.. autofunction:: ellipticity_intnorm
.. autofunction:: ellipticity_ryden04


Utilities
---------

.. autofunction:: triaxial_axis_ratio

'''

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def triaxial_axis_ratio(zeta, xi, size=None, *, rng=None):
    r'''axis ratio of a randomly projected triaxial ellipsoid

    Given the two axis ratios `1 >= zeta >= xi` of a randomly oriented triaxial
    ellipsoid, computes the axis ratio `q` of the projection.

    Parameters
    ----------
    zeta : array_like
        Axis ratio of intermediate and major axis.
    xi : array_like
        Axis ratio of minor and major axis.
    size : tuple of int or None
        Size of the random draw. If `None` is given, size is inferred from
        other inputs.
    rng : :class:`~numpy.random.Generator`, optional
        Random number generator.  If not given, a default RNG will be used.

    Returns
    -------
    q : array_like
        Axis ratio of the randomly projected ellipsoid.

    Notes
    -----
    See equations (11) and (12) in [1]_ for details.

    References
    ----------
    .. [1] Binney J., 1985, MNRAS, 212, 767. doi:10.1093/mnras/212.4.767

    '''

    # default RNG if not provided
    if rng is None:
        rng = np.random.default_rng()

    # get size from inputs if not explicitly provided
    if size is None:
        size = np.broadcast(zeta, xi).shape

    # draw random viewing angle (theta, phi)
    cos2_theta = rng.uniform(low=-1., high=1., size=size)
    cos2_theta *= cos2_theta
    sin2_theta = 1 - cos2_theta
    cos2_phi = np.cos(rng.uniform(low=0., high=2*np.pi, size=size))
    cos2_phi *= cos2_phi
    sin2_phi = 1 - cos2_phi

    # transform arrays to quantities that are used in eq. (11)
    z2m1 = np.square(zeta)
    z2m1 -= 1
    x2 = np.square(xi)

    # eq. (11) multiplied by xi^2 zeta^2
    A = (1 + z2m1*sin2_phi)*cos2_theta + x2*sin2_theta
    B2 = 4*z2m1**2*cos2_theta*sin2_phi*cos2_phi
    C = 1 + z2m1*cos2_phi

    # eq. (12)
    q = np.sqrt((A+C-np.sqrt((A-C)**2+B2))/(A+C+np.sqrt((A-C)**2+B2)))

    return q


def ellipticity_ryden04(mu, sigma, gamma, sigma_gamma, size=None, *, rng=None):
    r'''ellipticity distribution following Ryden (2004)

    The ellipticities are sampled by randomly projecting a 3D ellipsoid with
    principal axes :math:`A > B > C` [1]_.  The distribution of :math:`\log(1 -
    B/A)` is normal with mean :math:`\mu` and standard deviation :math:`\sigma`.
    The distribution of :math:`1 - C/B` is normal with mean :math:`\gamma` and
    standard deviation :math:`\sigma_\gamma` [2]_.  Both distributions are
    truncated to produce ratios in the range 0 to 1 using rejection sampling.

    Parameters
    ----------
    mu : array_like
        Mean of the truncated normal for :math:`\log(1 - B/A)`.
    sigma : array_like
        Standard deviation for :math:`\log(1 - B/A)`.
    gamma : array_like
        Mean of the truncated normal for :math:`1 - C/B`.
    sigma_gamma : array_like
        Standard deviation for :math:`1 - C/B`.
    size : int or tuple of ints or None
        Sample size.  If ``None``, the size is inferred from the parameters.
    rng : :class:`~numpy.random.Generator`, optional
        Random number generator.  If not given, a default RNG will be used.

    Returns
    -------
    eps : array_like
        Array of :term:`ellipticity` from projected axis ratios.

    References
    ----------
    .. [1] Ryden B. S., 2004, ApJ, 601, 214.
    .. [2] Padilla N. D., Strauss M. A., 2008, MNRAS, 388, 1321.

    '''

    # default RNG if not provided
    if rng is None:
        rng = np.random.default_rng()

    # draw gamma and epsilon from truncated normal -- eq.s (10)-(11)
    # first sample unbounded normal, then rejection sample truncation
    eps = rng.normal(mu, sigma, size=size)
    bad = (eps > 0)
    while np.any(bad):
        eps[bad] = rng.normal(mu, sigma, size=eps[bad].shape)
        bad = (eps > 0)
    gam = rng.normal(gamma, sigma_gamma, size=size)
    bad = (gam < 0) | (gam > 1)
    while np.any(bad):
        gam[bad] = rng.normal(gamma, sigma_gamma, size=gam[bad].shape)
        bad = (gam < 0) | (gam > 1)

    # compute triaxial axis ratios zeta = B/A, xi = C/A
    zeta = -np.expm1(eps)
    xi = (1 - gam)*zeta

    # random projection of random triaxial ellipsoid
    q = triaxial_axis_ratio(zeta, xi, rng=rng)

    # assemble ellipticity with random complex phase
    e = np.exp(1j*rng.uniform(0, 2*np.pi, size=np.shape(q)))
    e *= (1-q)/(1+q)

    # return the ellipticity
    return e


def ellipticity_gaussian(count: int | ArrayLike, sigma: ArrayLike, *,
                         rng: np.random.Generator | None = None
                         ) -> NDArray:
    r'''Sample Gaussian galaxy ellipticities.

    The ellipticities are sampled from a normal distribution with
    standard deviation ``sigma`` for each component.  Samples where the
    ellipticity is larger than unity are discarded.  Hence, do not use
    this function with too large values of ``sigma``, or the sampling
    will become inefficient.

    Parameters
    ----------
    count : int or array_like
        Number of ellipticities to be sampled.
    sigma : array_like
        Standard deviation in each component.
    rng : :class:`~numpy.random.Generator`, optional
        Random number generator.  If not given, a default RNG is used.

    Returns
    -------
    eps : array_like
        Array of galaxy :term:`ellipticity`.

    '''

    # default RNG if not provided
    if rng is None:
        rng = np.random.default_rng()

    # bring inputs into common shape
    count, sigma = np.broadcast_arrays(count, sigma)

    # allocate flattened output array
    eps = np.empty(count.sum(), dtype=np.complex128)

    # sample complex ellipticities
    # reject those where abs(e) > 0
    i = 0
    for k in np.ndindex(count.shape):
        e = rng.standard_normal(2*count[k], np.float64).view(np.complex128)
        e *= sigma[k]
        r = np.where(np.abs(e) > 1)[0]
        while len(r) > 0:
            rng.standard_normal(2*len(r), np.float64, e[r].view(np.float64))
            e[r] *= sigma[k]
            r = r[np.abs(e[r]) > 1]
        eps[i:i+count[k]] = e
        i += count[k]

    return eps


def ellipticity_intnorm(count: int | ArrayLike, sigma: ArrayLike, *,
                        rng: np.random.Generator | None = None
                        ) -> NDArray:
    r'''Sample galaxy ellipticities with intrinsic normal distribution.

    The ellipticities are sampled from an intrinsic normal distribution
    with standard deviation ``sigma`` for each component.

    Parameters
    ----------
    count : int | array_like
        Number of ellipticities to sample.
    sigma : array_like
        Standard deviation of the ellipticity in each component.
    rng : :class:`~numpy.random.Generator`, optional
        Random number generator.  If not given, a default RNG is used.

    Returns
    -------
    eps : array_like
        Array of galaxy :term:`ellipticity`.

    '''

    # default RNG if not provided
    if rng is None:
        rng = np.random.default_rng()

    # bring inputs into common shape
    count, sigma = np.broadcast_arrays(count, sigma)

    # make sure sigma is admissible
    if not np.all((0 <= sigma) & (sigma < 0.5**0.5)):
        raise ValueError('sigma must be between 0 and sqrt(0.5)')

    # convert to sigma_eta using fit
    sigma_eta = sigma*((8 + 5*sigma**2)/(2 - 4*sigma**2))**0.5

    # allocate flattened output array
    eps = np.empty(count.sum(), dtype=np.complex128)

    # sample complex ellipticities
    i = 0
    for k in np.ndindex(count.shape):
        e = rng.standard_normal(2*count[k], np.float64).view(np.complex128)
        e *= sigma_eta[k]
        r = np.hypot(e.real, e.imag)
        e *= np.divide(np.tanh(r/2), r, where=(r > 0), out=r)
        eps[i:i+count[k]] = e
        i += count[k]

    return eps
