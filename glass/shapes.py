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

"""  # noqa: D400

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import array_api_compat
import array_api_extra as xpx

from glass import _rng
from glass._array_api_utils import xp_additions as uxpx

if TYPE_CHECKING:
    from types import ModuleType

    from glass._types import ComplexArray, FloatArray, IntArray, UnifiedGenerator


def _populate_random_complex_array(
    length: int,
    rng: UnifiedGenerator,
) -> ComplexArray:
    return rng.standard_normal((length,)) + (1j * rng.standard_normal((length,)))


def triaxial_axis_ratio(
    zeta: float | FloatArray,
    xi: float | FloatArray,
    size: int | tuple[int, ...] | None = None,
    *,
    rng: UnifiedGenerator | None = None,
    xp: ModuleType | None = None,
) -> FloatArray:
    """
    Axis ratio of a randomly projected triaxial ellipsoid.

    Given the two axis ratios `1 >= zeta >= xi` of a randomly oriented triaxial
    ellipsoid, computes the axis ratio `q` of the projection.

    Parameters
    ----------
    zeta
        Axis ratio of intermediate and major axis.
    xi
        Axis ratio of minor and major axis.
    size
        Size of the random draw.
    rng
        Random number generator. If not given, a default RNG is used.
    xp
        The array library backend to use for array operations. If this is not
        specified, the backend will be determined from the input arrays.

    Returns
    -------
        The axis ratio of the randomly projected ellipsoid.

    Notes
    -----
    See equations (11) and (12) in [Binney85]_ for details.

    """
    if xp is None:
        xp = array_api_compat.array_namespace(zeta, xi, use_compat=False)

    zeta = xp.asarray(zeta)
    xi = xp.asarray(xi)

    # default RNG if not provided
    if rng is None:
        rng = _rng.rng_dispatcher(xp=xp)

    # get size from inputs if not explicitly provided
    if size is None:
        size = xp.broadcast_arrays(zeta, xi)[0].shape

    # draw random viewing angle (theta, phi)
    cos2_theta = rng.uniform(low=-1.0, high=1.0, size=size)
    cos2_theta *= cos2_theta
    sin2_theta = 1 - cos2_theta
    cos2_phi = xp.cos(rng.uniform(low=0.0, high=2 * math.pi, size=size))
    cos2_phi *= cos2_phi
    sin2_phi = 1 - cos2_phi

    # transform arrays to quantities that are used in eq. (11)
    z2m1 = xp.square(zeta)
    z2m1 -= 1
    x2 = xp.square(xi)

    # eq. (11) multiplied by xi^2 zeta^2
    a = (1 + z2m1 * sin2_phi) * cos2_theta + x2 * sin2_theta
    b2 = 4 * z2m1**2 * cos2_theta * sin2_phi * cos2_phi
    c = 1 + z2m1 * cos2_phi

    # eq. (12)
    return xp.sqrt(
        (a + c - xp.sqrt((a - c) ** 2 + b2)) / (a + c + xp.sqrt((a - c) ** 2 + b2)),
    )


def ellipticity_ryden04(  # noqa: PLR0913
    mu: float | FloatArray,
    sigma: float | FloatArray,
    gamma: float | FloatArray,
    sigma_gamma: float | FloatArray,
    size: int | tuple[int, ...] | None = None,
    *,
    rng: UnifiedGenerator | None = None,
    xp: ModuleType | None = None,
) -> FloatArray:
    r"""
    Ellipticity distribution following Ryden (2004).

    The ellipticities are sampled by randomly projecting a 3D ellipsoid with
    principal axes :math:`A > B > C` [Ryden04]_. The distribution of :math:`\log(1 -
    B/A)` is normal with mean :math:`\mu` and standard deviation :math:`\sigma`.
    The distribution of :math:`1 - C/B` is normal with mean :math:`\gamma` and
    standard deviation :math:`\sigma_\gamma` [Padilla08]_. Both distributions are
    truncated to produce ratios in the range 0 to 1 using rejection sampling.

    Parameters
    ----------
    mu
        Mean of the truncated normal for :math:`\log(1 - B/A)`.
    sigma
        Standard deviation for :math:`\log(1 - B/A)`.
    gamma
        Mean of the truncated normal for :math:`1 - C/B`.
    sigma_gamma
        Standard deviation for :math:`1 - C/B`.
    size
        Sample size.
    rng
        Random number generator. If not given, a default RNG is used.
    xp
        The array library backend to use for array operations. If this is not
        specified, the backend will be determined from the input arrays.

    Returns
    -------
        An array of :term:`ellipticity` from projected axis ratios.

    """
    if xp is None:
        xp = array_api_compat.array_namespace(
            mu,
            sigma,
            gamma,
            sigma_gamma,
            use_compat=False,
        )

    mu = xp.asarray(mu)
    sigma = xp.asarray(sigma)
    gamma = xp.asarray(gamma)
    sigma_gamma = xp.asarray(sigma_gamma)

    # default RNG if not provided
    if rng is None:
        rng = _rng.rng_dispatcher(xp=xp)

    # default size if not given
    if size is None:
        size = xp.broadcast_arrays(mu, sigma, gamma, sigma_gamma)[0].shape

    # broadcast all inputs to output shape
    # this makes it possible to efficiently resample later
    mu = xp.broadcast_to(mu, size)
    sigma = xp.broadcast_to(sigma, size)
    gamma = xp.broadcast_to(gamma, size)
    sigma_gamma = xp.broadcast_to(sigma_gamma, size)

    # draw gamma and epsilon from truncated normal -- eq.s (10)-(11)
    # first sample unbounded normal, then rejection sample truncation
    eps = rng.normal(mu, sigma, size=size)
    while xp.any(bad := eps > 0):
        eps = xpx.at(eps)[bad].set(rng.normal(mu[bad], sigma[bad]))
    gam = rng.normal(gamma, sigma_gamma, size=size)
    while xp.any(bad := (gam < 0) | (gam > 1)):
        gam = xpx.at(gam)[bad].set(rng.normal(gamma[bad], sigma_gamma[bad]))

    # compute triaxial axis ratios zeta = B/A, xi = C/A
    zeta = -xp.expm1(eps)
    xi = (1 - gam) * zeta

    # random projection of random triaxial ellipsoid
    q = triaxial_axis_ratio(zeta, xi, rng=rng)

    # assemble ellipticity with random complex phase
    e = xp.exp(1j * rng.uniform(0, 2 * math.pi, size=q.shape))
    e *= (1 - q) / (1 + q)

    # return the ellipticity
    return e


def ellipticity_gaussian(
    count: int | IntArray,
    sigma: float | FloatArray,
    *,
    rng: UnifiedGenerator | None = None,
    xp: ModuleType | None = None,
) -> ComplexArray:
    """
    Sample Gaussian galaxy ellipticities.

    The ellipticities are sampled from a normal distribution with
    standard deviation ``sigma`` for each component. Samples where the
    ellipticity is larger than unity are discarded. Hence, do not use
    this function with too large values of ``sigma``, or the sampling
    will become inefficient.

    Parameters
    ----------
    count
        Number of ellipticities to be sampled.
    sigma
        Standard deviation in each component.
    rng
        Random number generator. If not given, a default RNG is used.
    xp
        The array library backend to use for array operations. If this is not
        specified, the backend will be determined from the input arrays.

    Returns
    -------
        An array of galaxy :term:`ellipticity`.

    """
    if xp is None:
        xp = array_api_compat.array_namespace(count, sigma, use_compat=False)
    # bring inputs into common shape
    count_broadcasted, sigma_broadcasted = xp.broadcast_arrays(
        xp.asarray(count),
        xp.asarray(sigma),
    )

    # default RNG if not provided
    if rng is None:
        rng = _rng.rng_dispatcher(xp=xp)

    # allocate flattened output array
    eps = xp.empty(xp.sum(count_broadcasted), dtype=xp.complex128)

    # sample complex ellipticities
    # reject those where abs(e) > 0
    i = 0
    for k in uxpx.ndindex(count_broadcasted.shape, xp=xp):
        e = _populate_random_complex_array(count_broadcasted[k], rng)
        e *= sigma_broadcasted[k]
        r = xp.abs(e) > 1
        while xp.count_nonzero(r) > 0:
            e = xpx.at(e)[r].set(
                _populate_random_complex_array(xp.count_nonzero(r), rng),
            )
            e = xpx.at(e)[r].multiply(sigma_broadcasted[k])
            r = xp.abs(e) > 1
        eps = xpx.at(eps)[i : i + count_broadcasted[k]].set(e)
        i += count_broadcasted[k]

    return eps  # ty: ignore[invalid-return-type]


def ellipticity_intnorm(
    count: int | IntArray,
    sigma: float | FloatArray,
    *,
    rng: UnifiedGenerator | None = None,
    xp: ModuleType | None = None,
) -> ComplexArray:
    """
    Sample galaxy ellipticities with intrinsic normal distribution.

    The ellipticities are sampled from an intrinsic normal distribution
    with standard deviation ``sigma`` for each component.

    Parameters
    ----------
    count
        Number of ellipticities to sample.
    sigma
        Standard deviation of the ellipticity in each component.
    rng
        Random number generator. If not given, a default RNG is used.
    xp
        The array library backend to use for array operations. If this is not
        specified, the backend will be determined from the input arrays.

    Returns
    -------
        An array of galaxy :term:`ellipticity`.

    Raises
    ------
    ValueError
        If the standard deviation is not in the range [0, sqrt(0.5)].

    """
    if xp is None:
        xp = array_api_compat.array_namespace(count, sigma, use_compat=False)
    # default RNG if not provided
    if rng is None:
        rng = _rng.rng_dispatcher(xp=xp)

    # bring inputs into common shape
    count_broadcasted, sigma_broadcasted = xp.broadcast_arrays(
        xp.asarray(count),
        xp.asarray(sigma),
    )

    # make sure sigma is admissible
    if not xp.all((sigma_broadcasted >= 0) & (sigma_broadcasted < 0.5**0.5)):
        msg = "sigma must be between 0 and sqrt(0.5)"
        raise ValueError(msg)

    # convert to sigma_eta using fit
    sigma_eta = (
        sigma_broadcasted
        * ((8 + 5 * sigma_broadcasted**2) / (2 - 4 * sigma_broadcasted**2)) ** 0.5
    )

    # allocate flattened output array
    eps = xp.empty(xp.sum(count_broadcasted), dtype=xp.complex128)

    # sample complex ellipticities
    i = 0
    for k in uxpx.ndindex(count_broadcasted.shape, xp=xp):
        e = _populate_random_complex_array(count_broadcasted[k], rng)
        e *= sigma_eta[k]
        r = xp.hypot(xp.real(e), xp.imag(e))
        e *= xp.where(
            r > 0,
            xp.divide(xp.tanh(r / 2), r),
            xp.asarray(1.0, dtype=e.dtype),
        )

        eps = xpx.at(eps)[i : i + count_broadcasted[k]].set(e)
        i += count_broadcasted[k]

    return eps  # ty: ignore[invalid-return-type]
