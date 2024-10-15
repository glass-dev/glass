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

"""  # noqa: D205, D400

from __future__ import annotations

import typing
import warnings

import healpix
import numpy as np
import numpy.typing as npt

from glass.core.array import broadcast_leading_axes, cumtrapz

if typing.TYPE_CHECKING:
    from glass.shells import RadialWindow


def redshifts(
    n: int | npt.NDArray[typing.Any],
    w: RadialWindow,
    *,
    rng: np.random.Generator | None = None,
) -> npt.NDArray[typing.Any]:
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
    return redshifts_from_nz(n, w.za, w.wa, rng=rng, warn=False)  # type: ignore[arg-type]


def redshifts_from_nz(
    count: int | npt.NDArray[typing.Any],
    z: npt.NDArray[typing.Any],
    nz: npt.NDArray[typing.Any],
    *,
    rng: np.random.Generator | None = None,
    warn: bool = True,
) -> npt.NDArray[typing.Any]:
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
    dims, count, z, nz = broadcast_leading_axes((count, 0), (z, 1), (nz, 1))  # type: ignore[arg-type, assignment]

    # list of results for all dimensions
    redshifts = np.empty(count.sum())  # type: ignore[union-attr]

    # keep track of the number of sampled redshifts
    total = 0

    # go through extra dimensions; also works if dims is empty
    for k in np.ndindex(dims):
        # compute the CDF of each galaxy population
        cdf = cumtrapz(nz[k], z[k], dtype=float)  # type: ignore[arg-type]
        cdf /= cdf[-1]

        # sample redshifts and store result
        redshifts[total : total + count[k]] = np.interp(  # type: ignore[index]
            rng.uniform(0, 1, size=count[k]),  # type: ignore[index]
            cdf,
            z[k],
        )
        total += count[k]  # type: ignore[index]

    assert total == redshifts.size  # noqa: S101

    return redshifts


def galaxy_shear(  # noqa: PLR0913
    lon: npt.NDArray[typing.Any],
    lat: npt.NDArray[typing.Any],
    eps: npt.NDArray[typing.Any],
    kappa: npt.NDArray[typing.Any],
    gamma1: npt.NDArray[typing.Any],
    gamma2: npt.NDArray[typing.Any],
    *,
    reduced_shear: bool = True,
) -> npt.NDArray[typing.Any]:
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
    z: npt.NDArray[typing.Any],
    sigma_0: float | npt.NDArray[typing.Any],
    *,
    lower: npt.NDArray[typing.Any] | None = None,
    upper: npt.NDArray[typing.Any] | None = None,
    rng: np.random.Generator | None = None,
) -> npt.NDArray[typing.Any]:
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
        lower = 0.0  # type: ignore[assignment]
    if upper is None:
        upper = np.inf  # type: ignore[assignment]

    if not np.all(lower < upper):  # type: ignore[operator]
        msg = "requires lower < upper"
        raise ValueError(msg)

    if not dims:
        while zphot < lower or zphot > upper:  # type: ignore[operator]
            zphot = rng.normal(z, sigma)
    else:
        z = np.broadcast_to(z, dims)
        trunc = np.where((zphot < lower) | (zphot > upper))[0]  # type: ignore[operator]
        while trunc.size:
            znew = rng.normal(z[trunc], sigma[trunc])
            zphot[trunc] = znew
            trunc = trunc[(znew < lower) | (znew > upper)]  # type: ignore[operator]

    return zphot
