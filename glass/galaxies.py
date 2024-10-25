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
    n: int | npt.NDArray[np.float64],
    w: RadialWindow,
    *,
    rng: np.random.Generator | None = None,
) -> npt.NDArray[np.float64]:
    """
    _summary_.

    Parameters
    ----------
    n
        _description_
    w
        _description_
    rng
        _description_

    Returns
    -------
        Random redshifts following the radial window function.

    """
    return redshifts_from_nz(n, w.za, w.wa, rng=rng, warn=False)


def redshifts_from_nz(
    count: int | npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
    nz: npt.NDArray[np.float64],
    *,
    rng: np.random.Generator | None = None,
    warn: bool = True,
) -> npt.NDArray[np.float64]:
    """
    _summary_.

    Parameters
    ----------
    count
        _description_
    z
        _description_
    nz
        _description_
    rng
        _description_
    warn
        _description_

    Returns
    -------
        Redshifts sampled from the given source distribution. For
        inputs with extra dimensions, returns a flattened 1-D array of
        samples from all populations.

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
    dims, *rest = broadcast_leading_axes((count, 0), (z, 1), (nz, 1))
    count_out, z_out, nz_out = rest

    # list of results for all dimensions
    redshifts = np.empty(count_out.sum())

    # keep track of the number of sampled redshifts
    total = 0

    # go through extra dimensions; also works if dims is empty
    for k in np.ndindex(dims):
        # compute the CDF of each galaxy population
        cdf = cumtrapz(nz_out[k], z_out[k], dtype=float)
        cdf /= cdf[-1]

        # sample redshifts and store result
        redshifts[total : total + count_out[k]] = np.interp(
            rng.uniform(0, 1, size=count_out[k]),
            cdf,
            z_out[k],
        )
        total += count_out[k]

    assert total == redshifts.size  # noqa: S101

    return redshifts


def galaxy_shear(  # noqa: PLR0913
    lon: npt.NDArray[np.float64],
    lat: npt.NDArray[np.float64],
    eps: npt.NDArray[np.float64],
    kappa: npt.NDArray[np.float64],
    gamma1: npt.NDArray[np.float64],
    gamma2: npt.NDArray[np.float64],
    *,
    reduced_shear: bool = True,
) -> npt.NDArray[np.float64]:
    """
    _summary_.

    Parameters
    ----------
    lon
        _description_
    lat
        _description_
    eps
        _description_
    kappa
        _description_
    gamma1
        _description_
    gamma2
        _description_
    reduced_shear
        _description_

    Returns
    -------
        An array of complex-valued observed galaxy shears
        (lensed ellipticities).

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
    """
    _summary_.

    Parameters
    ----------
    z
        _description_
    sigma_0
        _description_
    lower
        _description_
    upper
        _description_
    rng
        _description_

    Returns
    -------
        Photometric redshifts assuming Gaussian errors, of the same
        shape as *z*.

    Raises
    ------
    ValueError
        _description_

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
    """
    _summary_.

    Parameters
    ----------
    delta
        _description_
    zeff
        _description_
    a_ia
        _description_
    cosmo
        _description_
    z0
        _description_
    eta
        _description_
    lbar
        _description_
    l0
        _description_
    beta
        _description_

    Returns
    -------
        The effective convergence due to intrinsic alignments.

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
