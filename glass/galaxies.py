"""
Galaxies
========

.. currentmodule:: glass

The following functions provide functionality for simulating galaxies
as typically observed in a cosmological galaxy survey.

Functions
---------

.. autofunction:: redshifts
.. autofunction:: redshifts_from_nz
.. autofunction:: galaxy_shear
.. autofunction:: gaussian_phz

Intrinsic alignments
--------------------
.. autofunction:: kappa_ia_nla

"""  # noqa: D205, D400

from __future__ import annotations

import typing
import warnings

import healpix
import numpy as np
import numpy.typing as npt

from glass.core.array import broadcast_leading_axes, cumtrapz

if typing.TYPE_CHECKING:
    from cosmology import Cosmology

    from glass.shells import RadialWindow


def redshifts(
    n: int | list[int] | list[list[int]],
    w: RadialWindow,
    *,
    rng: np.random.Generator | None = None,
) -> npt.NDArray[np.float64]:
    """
    Sample redshifts from a radial window function.

    This function samples *n* redshifts from a distribution that follows
    the given radial window function *w*.

    Returns random redshifts following the radial window function.

    Parameters
    ----------
    n:
        Number of redshifts to sample. If an array is given, the
        results are concatenated.
    w:
        Radial window function.
    rng:
        Random number generator. If not given, a default RNG is used.

    """
    return redshifts_from_nz(n, w.za, w.wa, rng=rng, warn=False)


def redshifts_from_nz(
    count: float
    | list[float]
    | list[int]
    | list[list[int]]
    | list[npt.NDArray[np.float64]]
    | npt.NDArray[np.float64]
    | npt.NDArray[np.int_],
    z: list[float] | npt.NDArray[np.float64] | npt.NDArray[np.int_],
    nz: list[int]
    | list[float]
    | npt.NDArray[np.float64]
    | list[npt.NDArray[np.float64]]
    | npt.NDArray[np.int_],
    *,
    rng: np.random.Generator | None = None,
    warn: bool = True,
) -> npt.NDArray[np.float64]:
    """
    Generate galaxy redshifts from a source distribution.

    The function supports sampling from multiple populations of
    redshifts if *count* is an array or if there are additional axes in
    the *z* or *nz* arrays. In this case, the shape of *count* and the
    leading dimensions of *z* and *nz* are broadcast to a common shape,
    and redshifts are sampled independently for each extra dimension.
    The results are concatenated into a flat array.

    Returns redshifts sampled from the given source distribution. For
    inputs with extra dimensions, returns a flattened 1-D array of
    samples from all populations.

    Parameters
    ----------
    count:
        Number of redshifts to sample. If an array is given, its shape
        is broadcast against the leading axes of *z* and *nz*.
    z:
        Source distribution. Leading axes are broadcast against the
        shape of *count*.
    nz:
        Source distribution. Leading axes are broadcast against the
        shape of *count*.
    rng:
        Random number generator. If not given, a default RNG is used.
    warn:
        Throw relevant warnings.

    """
    if warn:
        warnings.warn(
            "when sampling galaxies, redshifts_from_nz() is often not the function you"
            " want. Try redshifts() instead. Use warn=False to suppress this warning.",
            stacklevel=2,
        )

    # get default RNG if not given
    if rng is None:
        rng = np.random.default_rng()

    # bring inputs' leading axes into common shape
    dims, count, z, nz = broadcast_leading_axes((count, 0), (z, 1), (nz, 1))  # type: ignore[arg-type, misc]

    # list of results for all dimensions
    redshifts = np.empty(count.sum())  # type: ignore[union-attr]

    # keep track of the number of sampled redshifts
    total = 0

    # go through extra dimensions; also works if dims is empty
    for k in np.ndindex(dims):
        # compute the CDF of each galaxy population
        cdf = cumtrapz(nz[k], z[k], dtype=float)  # type: ignore[call-overload]
        cdf /= cdf[-1]

        # sample redshifts and store result
        redshifts[total : total + count[k]] = np.interp(  # type: ignore[call-overload, index]
            rng.uniform(0, 1, size=count[k]),  # type: ignore[call-overload, index]
            cdf,
            z[k],  # type: ignore[call-overload]
        )
        total += count[k]  # type: ignore[call-overload, index]

    assert total == redshifts.size  # noqa: S101

    return redshifts


def galaxy_shear(  # noqa: PLR0913
    lon: list[float] | npt.NDArray[np.float64],
    lat: list[float] | npt.NDArray[np.float64],
    eps: list[float] | npt.NDArray[np.float64],
    kappa: npt.NDArray[np.float64],
    gamma1: npt.NDArray[np.float64],
    gamma2: npt.NDArray[np.float64],
    *,
    reduced_shear: bool = True,
) -> npt.NDArray[np.float64]:
    """
    Observed galaxy shears from weak lensing.

    Takes lensing maps for convergence and shear and produces a lensed
    ellipticity (shear) for each intrinsic galaxy ellipticity.

    Returns an array of complex-valued observed galaxy shears
    (lensed ellipticities).

    Parameters
    ----------
    lon:
        Array for galaxy longitudes.
    lat:
        Array for galaxy latitudes.
    eps:
        Array of galaxy :term:`ellipticity`.
    kappa:
        HEALPix map for convergence.
    gamma1:
        HEALPix maps for a component of shear.
    gamma2:
        HEALPix maps for a component of shear.
    reduced_shear:
        If ``False``, galaxy shears are not reduced
        by the convergence. Default is ``True``.

    """
    nside = healpix.npix2nside(np.broadcast(kappa, gamma1, gamma2).shape[-1])

    size = np.broadcast(lon, lat, eps).size

    # output arrays
    k = np.empty(size)
    g = np.empty(size, dtype=complex)

    # get the lensing maps at galaxy position
    for i in range(0, size, 10000):
        s = slice(i, i + 10000)
        ipix = healpix.ang2pix(nside, lon[s], lat[s], lonlat=True)
        k[s] = kappa[ipix]
        g.real[s] = gamma1[ipix]
        g.imag[s] = gamma2[ipix]

    if reduced_shear:
        # compute reduced shear in place
        g /= 1 - k

        # compute lensed ellipticities
        g = (eps + g) / (1 + g.conj() * eps)
    else:
        # simple sum of shears
        g += eps

    return g


def gaussian_phz(
    z: float | npt.NDArray[np.float64],
    sigma_0: float | npt.NDArray[np.float64],
    *,
    lower: float | npt.NDArray[np.float64] | None = None,
    upper: float | npt.NDArray[np.float64] | None = None,
    rng: np.random.Generator | None = None,
) -> float | npt.NDArray[np.float64]:
    r"""
    Photometric redshifts assuming a Gaussian error.

    A simple toy model of photometric redshift errors that assumes a
    Gaussian error with redshift-dependent standard deviation
    :math:`\sigma(z) = (1 + z) \sigma_0` [1].

    Returns photometric redshifts assuming Gaussian errors, of the same
    shape as *z*.

    Parameters
    ----------
    z:
        True redshifts.
    sigma_0:
        Redshift error in the tomographic binning at zero redshift.
    lower:
        Bounds for the returned photometric redshifts.
    upper:
        Bounds for the returned photometric redshifts.
    rng:
        Random number generator. If not given, a default RNG is used.

    Warnings
    --------
    The *lower* and *upper* bounds are implemented using plain rejection
    sampling from the non-truncated normal distribution. If bounds are
    used, they should always contain significant probability mass.

    See Also
    --------
    glass.tomo_nz_gausserr:
        Create tomographic redshift distributions assuming the same model.

    References
    ----------
    * [1] Amara A., Réfrégier A., 2007, MNRAS, 381, 1018.
           doi:10.1111/j.1365-2966.2007.12271.x

    Examples
    --------
    See the :doc:`/examples/1-basic/photoz` example.

    """
    # get default RNG if not given
    if rng is None:
        rng = np.random.default_rng()

    sigma = np.add(1, z) * sigma_0
    dims = np.shape(sigma)

    zphot = rng.normal(z, sigma)

    if lower is None:
        lower = 0.0
    if upper is None:
        upper = np.inf

    if not np.all(lower < upper):
        msg = "requires lower < upper"
        raise ValueError(msg)

    if not dims:
        while zphot < lower or zphot > upper:
            zphot = rng.normal(z, sigma)
    else:
        z = np.broadcast_to(z, dims)
        trunc = np.where((zphot < lower) | (zphot > upper))[0]
        while trunc.size:
            znew = rng.normal(z[trunc], sigma[trunc])
            zphot[trunc] = znew
            trunc = trunc[(znew < lower) | (znew > upper)]

    return zphot


def kappa_ia_nla(  # noqa: PLR0913
    delta: npt.NDArray[np.float64],
    zeff: float,
    a_ia: float,
    cosmo: Cosmology,
    *,
    z0: float = 0.0,
    eta: float = 0.0,
    lbar: float = 0.0,
    l0: float = 1e-9,
    beta: float = 0.0,
) -> npt.NDArray[np.float64]:
    r"""
    Effective convergence from intrinsic alignments using the NLA model.

    Returns the effective convergence due to intrinsic alignments.

    Parameters
    ----------
    delta:
        Matter density contrast.
    zeff:
        Effective redshift of the matter field.
    a_ia:
        Intrinsic alignments amplitude.
    cosmo:
        Cosmology instance.
    z0:
        Reference redshift for the redshift dependence.
    eta:
        Power of the redshift dependence.
    lbar:
        Mean luminosity of the galaxy sample.
    l0:
        Reference luminosity for the luminosity dependence.
    beta:
        Power of the luminosity dependence.

    Notes
    -----
    The Non-linear Alignments Model (NLA) describes an effective
    convergence :math:`\kappa_{\rm IA}` that models the effect of
    intrinsic alignments.  It is computed from the matter density
    contrast :math:`\delta` as [1] [3]

    .. math::

        \kappa_{\rm IA} = f_{\rm NLA} \, \delta \;,

    where the NLA factor :math:`f_{\rm NLA}` is defined as [4] [5]

    .. math::

        f_{\rm{NLA}}
        = -A_{\rm IA} \, \frac{C_1 \, \bar{\rho}(z)}{D(z)} \,
            \biggl(\frac{1+z}{1+z_0}\biggr)^\eta \,
            \biggl(\frac{\bar{L}}{L_0}\biggr)^\beta \;,

    with

    * :math:`A_{\rm IA}` the intrinsic alignments amplitude,
    * :math:`C_1` a normalisation constant [2],
    * :math:`z` the effective redshift of the model,
    * :math:`\bar{\rho}` the mean matter density,
    * :math:`D` the growth factor,
    * :math:`\eta` the power that describes the redshift-dependence with
      respect to :math:`z_0`,
    * :math:`\bar{L}` the mean luminosity of the galaxy sample, and
    * :math:`\beta` the power that describes the luminosity-dependence
      :math:`\bar{L}` with respect to :math:`L_0`.

    References
    ----------
    * [1] Catelan P., Kamionkowski M., Blandford R. D., 2001, MNRAS,
       320, L7. doi:10.1046/j.1365-8711.2001.04105.x
    * [2] Hirata C. M., Seljak U., 2004, PhRvD, 70, 063526.
       doi:10.1103/PhysRevD.70.063526
    * [3] Bridle S., King L., 2007, NJPh, 9, 444.
       doi:10.1088/1367-2630/9/12/444
    * [4] Johnston, H., Georgiou, C., Joachimi, B., et al., 2019,
        A&A, 624, A30. doi:10.1051/0004-6361/201834714
    * [5] Tessore, N., Loureiro, A., Joachimi, B., et al., 2023,
       OJAp, 6, 11. doi:10.21105/astro.2302.01942

    """
    c1 = 5e-14 / cosmo.h**2  # Solar masses per cubic Mpc
    rho_c1 = c1 * cosmo.rho_c0

    prefactor = -a_ia * rho_c1 * cosmo.Om
    inverse_linear_growth = 1.0 / cosmo.gf(zeff)
    redshift_dependence = ((1 + zeff) / (1 + z0)) ** eta
    luminosity_dependence = (lbar / l0) ** beta

    f_nla = (
        prefactor * inverse_linear_growth * redshift_dependence * luminosity_dependence
    )

    return delta * f_nla
