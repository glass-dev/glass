"""
Observed shapes
===============

.. currentmodule:: glass

The following functions provide functionality for simulating the
observed shapes of objects, such as e.g. galaxies.

Ellipticity
-----------

.. autofunction:: ellipticity_gaussian
.. autofunction:: ellipticity_intnorm
.. autofunction:: ellipticity_ryden04


Utilities
---------

.. autofunction:: triaxial_axis_ratio

"""  # noqa: D205, D400

from __future__ import annotations

import typing

import numpy as np
import numpy.typing as npt


def triaxial_axis_ratio(zeta, xi, size=None, *, rng=None):  # type: ignore[no-untyped-def]
    r"""
    Axis ratio of a randomly projected triaxial ellipsoid.

    Given the two axis ratios `1 >= zeta >= xi` of a randomly oriented triaxial
    ellipsoid, computes the axis ratio `q` of the projection.

    Returns the axis ratio of the randomly projected ellipsoid.

    Parameters
    ----------
    zeta:
        Axis ratio of intermediate and major axis.
    xi:
        Axis ratio of minor and major axis.
    size:
        Size of the random draw. If `None` is given, size is inferred from
        other inputs.
    rng:
        Random number generator. If not given, a default RNG will be used.

    Notes
    -----
    See equations (11) and (12) in [1] for details.

    References
    ----------
    * [1] Binney J., 1985, MNRAS, 212, 767. doi:10.1093/mnras/212.4.767

    """
    # default RNG if not provided
    if rng is None:
        rng = np.random.default_rng()

    # get size from inputs if not explicitly provided
    if size is None:
        size = np.broadcast(zeta, xi).shape

    # draw random viewing angle (theta, phi)
    cos2_theta = rng.uniform(low=-1.0, high=1.0, size=size)
    cos2_theta *= cos2_theta
    sin2_theta = 1 - cos2_theta
    cos2_phi = np.cos(rng.uniform(low=0.0, high=2 * np.pi, size=size))
    cos2_phi *= cos2_phi
    sin2_phi = 1 - cos2_phi

    # transform arrays to quantities that are used in eq. (11)
    z2m1 = np.square(zeta)
    z2m1 -= 1
    x2 = np.square(xi)

    # eq. (11) multiplied by xi^2 zeta^2
    A = (1 + z2m1 * sin2_phi) * cos2_theta + x2 * sin2_theta  # noqa: N806
    B2 = 4 * z2m1**2 * cos2_theta * sin2_phi * cos2_phi  # noqa: N806
    C = 1 + z2m1 * cos2_phi  # noqa: N806

    # eq. (12)
    return np.sqrt(
        (A + C - np.sqrt((A - C) ** 2 + B2)) / (A + C + np.sqrt((A - C) ** 2 + B2)),
    )


def ellipticity_ryden04(mu, sigma, gamma, sigma_gamma, size=None, *, rng=None):  # type: ignore[no-untyped-def] # noqa: PLR0913
    r"""
    Ellipticity distribution following Ryden (2004).

    The ellipticities are sampled by randomly projecting a 3D ellipsoid with
    principal axes :math:`A > B > C` [1]. The distribution of :math:`\log(1 -
    B/A)` is normal with mean :math:`\mu` and standard deviation :math:`\sigma`.
    The distribution of :math:`1 - C/B` is normal with mean :math:`\gamma` and
    standard deviation :math:`\sigma_\gamma` [2]. Both distributions are
    truncated to produce ratios in the range 0 to 1 using rejection sampling.

    Returns an array of :term:`ellipticity` from projected axis ratios.

    Parameters
    ----------
    mu:
        Mean of the truncated normal for :math:`\log(1 - B/A)`.
    sigma:
        Standard deviation for :math:`\log(1 - B/A)`.
    gamma:
        Mean of the truncated normal for :math:`1 - C/B`.
    sigma_gamma:
        Standard deviation for :math:`1 - C/B`.
    size:
        Sample size. If ``None``, the size is inferred from the parameters.
    rng:
        Random number generator. If not given, a default RNG will be used.

    References
    ----------
    * [1] Ryden B. S., 2004, ApJ, 601, 214.
    * [2] Padilla N. D., Strauss M. A., 2008, MNRAS, 388, 1321.

    """
    # default RNG if not provided
    if rng is None:
        rng = np.random.default_rng()

    # default size if not given
    if size is None:
        size = np.broadcast(mu, sigma, gamma, sigma_gamma).shape

    # broadcast all inputs to output shape
    # this makes it possible to efficiently resample later
    mu = np.broadcast_to(mu, size, subok=True)
    sigma = np.broadcast_to(sigma, size, subok=True)
    gamma = np.broadcast_to(gamma, size, subok=True)
    sigma_gamma = np.broadcast_to(sigma_gamma, size, subok=True)

    # draw gamma and epsilon from truncated normal -- eq.s (10)-(11)
    # first sample unbounded normal, then rejection sample truncation
    eps = rng.normal(mu, sigma, size=size)
    while np.any(bad := eps > 0):
        eps[bad] = rng.normal(mu[bad], sigma[bad])
    gam = rng.normal(gamma, sigma_gamma, size=size)
    while np.any(bad := (gam < 0) | (gam > 1)):
        gam[bad] = rng.normal(gamma[bad], sigma_gamma[bad])

    # compute triaxial axis ratios zeta = B/A, xi = C/A
    zeta = -np.expm1(eps)
    xi = (1 - gam) * zeta

    # random projection of random triaxial ellipsoid
    q = triaxial_axis_ratio(zeta, xi, rng=rng)  # type: ignore[no-untyped-call]

    # assemble ellipticity with random complex phase
    e = np.exp(1j * rng.uniform(0, 2 * np.pi, size=np.shape(q)))
    e *= (1 - q) / (1 + q)

    # return the ellipticity
    return e


def ellipticity_gaussian(
    count: int | npt.ArrayLike,
    sigma: npt.ArrayLike,
    *,
    rng: np.random.Generator | None = None,
) -> npt.NDArray[typing.Any]:
    r"""
    Sample Gaussian galaxy ellipticities.

    The ellipticities are sampled from a normal distribution with
    standard deviation ``sigma`` for each component. Samples where the
    ellipticity is larger than unity are discarded. Hence, do not use
    this function with too large values of ``sigma``, or the sampling
    will become inefficient.

    Returns an array of galaxy :term:`ellipticity`.

    Parameters
    ----------
    count:
        Number of ellipticities to be sampled.
    sigma:
        Standard deviation in each component.
    rng:
        Random number generator. If not given, a default RNG is used.

    """
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
        e = rng.standard_normal(2 * count[k], np.float64).view(np.complex128)
        e *= sigma[k]
        r = np.where(np.abs(e) > 1)[0]
        while len(r) > 0:
            rng.standard_normal(2 * len(r), np.float64, e[r].view(np.float64))
            e[r] *= sigma[k]
            r = r[np.abs(e[r]) > 1]
        eps[i : i + count[k]] = e
        i += count[k]

    return eps


def ellipticity_intnorm(
    count: int | npt.ArrayLike,
    sigma: npt.ArrayLike,
    *,
    rng: np.random.Generator | None = None,
) -> npt.NDArray[typing.Any]:
    r"""
    Sample galaxy ellipticities with intrinsic normal distribution.

    The ellipticities are sampled from an intrinsic normal distribution
    with standard deviation ``sigma`` for each component.

    Returns an array of galaxy :term:`ellipticity`.

    Parameters
    ----------
    count:
        Number of ellipticities to sample.
    sigma:
        Standard deviation of the ellipticity in each component.
    rng:
        Random number generator. If not given, a default RNG is used.

    """
    # default RNG if not provided
    if rng is None:
        rng = np.random.default_rng()

    # bring inputs into common shape
    count, sigma = np.broadcast_arrays(count, sigma)

    # make sure sigma is admissible
    if not np.all((sigma >= 0) & (sigma < 0.5**0.5)):
        msg = "sigma must be between 0 and sqrt(0.5)"
        raise ValueError(msg)

    # convert to sigma_eta using fit
    sigma_eta = sigma * ((8 + 5 * sigma**2) / (2 - 4 * sigma**2)) ** 0.5

    # allocate flattened output array
    eps = np.empty(count.sum(), dtype=np.complex128)

    # sample complex ellipticities
    i = 0
    for k in np.ndindex(count.shape):
        e = rng.standard_normal(2 * count[k], np.float64).view(np.complex128)
        e *= sigma_eta[k]
        r = np.hypot(e.real, e.imag)
        e *= np.divide(np.tanh(r / 2), r, where=(r > 0), out=r)
        eps[i : i + count[k]] = e
        i += count[k]

    return eps
