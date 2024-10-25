"""
Lensing
=======

.. currentmodule:: glass

The following functions/classes provide functionality for simulating
gravitational lensing by the matter distribution in the universe.

Iterative lensing
-----------------

.. autoclass:: MultiPlaneConvergence
.. autofunction:: multi_plane_matrix
.. autofunction:: multi_plane_weights


Lensing fields
--------------

.. autofunction:: from_convergence
.. autofunction:: shear_from_convergence


Applying lensing
----------------

.. autofunction:: deflect

"""  # noqa: D205, D400

from __future__ import annotations

import typing

import healpy as hp
import numpy as np
import numpy.typing as npt

if typing.TYPE_CHECKING:
    from cosmology import Cosmology

    from glass.shells import RadialWindow


def from_convergence(  # noqa: PLR0913
    kappa: npt.NDArray[np.float64],
    lmax: int | None = None,
    *,
    potential: bool = False,
    deflection: bool = False,
    shear: bool = False,
    discretized: bool = True,
) -> tuple[npt.NDArray[np.float64], ...]:
    r"""
    _summary_.

    Parameters
    ----------
    kappa
        _description_
    lmax
        _description_
    potential
        _description_
    deflection
        _description_
    shear
        _description_
    discretized
        _description_

    Returns
    -------
        The maps of:

        * deflection potential if ``potential`` is true.
        * potential (complex) if ``deflection`` is true.
        * shear (complex) if ``shear`` is true.

    Notes
    -----
    The weak lensing fields are computed from the convergence or
    deflection potential in the following way. [1]

    Define the spin-raising and spin-lowering operators of the
    spin-weighted spherical harmonics as

    .. math::

        \eth {}_sY_{lm}
        = +\sqrt{(l-s)(l+s+1)} \, {}_{s+1}Y_{lm} \;, \\
        \bar{\eth} {}_sY_{lm}
        = -\sqrt{(l+s)(l-s+1)} \, {}_{s-1}Y_{lm} \;.

    The convergence field :math:`\kappa` is related to the deflection
    potential field :math:`\psi` by the Poisson equation,

    .. math::

        2 \kappa
        = \eth\bar{\eth} \, \psi
        = \bar{\eth}\eth \, \psi \;.

    The convergence modes :math:`\kappa_{lm}` are hence related to the
    deflection potential modes :math:`\psi_{lm}` as

    .. math::

        2 \kappa_{lm}
        = -l \, (l+1) \, \psi_{lm} \;.

    The :term:`deflection` :math:`\alpha` is the gradient of the
    deflection potential :math:`\psi`. On the sphere, this is

    .. math::

        \alpha
        = \eth \, \psi \;.

    The deflection field has spin weight :math:`1` in the HEALPix
    convention, in order for points to be deflected towards regions of
    positive convergence. The modes :math:`\alpha_{lm}` of the
    deflection field are hence

    .. math::
        \alpha_{lm}
        = \sqrt{l \, (l+1)} \, \psi_{lm} \;.

    The shear field :math:`\gamma` is related to the deflection
    potential :math:`\psi` and deflection :math:`\alpha` as

    .. math::
        2 \gamma
        = \eth\eth \, \psi
        = \eth \, \alpha \;,

    and thus has spin weight :math:`2`. The shear modes
    :math:`\gamma_{lm}` are related to the deflection potential modes as

    .. math::
        2 \gamma_{lm}
        = \sqrt{(l+2) \, (l+1) \, l \, (l-1)} \, \psi_{lm} \;.

    References
    ----------
    * [1] Tessore N., et al., OJAp, 6, 11 (2023).
          doi:10.21105/astro.2302.01942

    """
    # no output means no computation, return empty tuple
    if not (potential or deflection or shear):
        return ()

    # get the NSIDE parameter
    nside = hp.get_nside(kappa)
    if lmax is None:
        lmax = 3 * nside - 1

    # compute alm
    alm = hp.map2alm(kappa, lmax=lmax, pol=False, use_pixel_weights=True)

    # mode number; all conversions are factors of this
    l = np.arange(lmax + 1)  # noqa: E741

    # this tuple will be returned
    results: tuple[npt.NDArray[np.float64], ...] = ()

    # convert convergence to potential
    fl = np.divide(-2, l * (l + 1), where=(l > 0), out=np.zeros(lmax + 1))
    hp.almxfl(alm, fl, inplace=True)

    # if potential is requested, compute map and add to output
    if potential:
        psi = hp.alm2map(alm, nside, lmax=lmax)
        results += (psi,)

    # if no spin-weighted maps are requested, stop here
    if not (deflection or shear):
        return results

    # zero B-modes for spin-weighted maps
    blm = np.zeros_like(alm)

    # compute deflection alms in place
    fl = np.sqrt(l * (l + 1))
    # TODO(ntessore): missing spin-1 pixel window function here # noqa: FIX002
    # https://github.com/glass-dev/glass/issues/243
    hp.almxfl(alm, fl, inplace=True)

    # if deflection is requested, compute spin-1 maps and add to output
    if deflection:
        alpha = hp.alm2map_spin([alm, blm], nside, 1, lmax)
        alpha = alpha[0] + 1j * alpha[1]
        results += (alpha,)

    # if no shear is requested, stop here
    if not shear:
        return results

    # compute shear alms in place
    # if discretised, factor out spin-0 kernel and apply spin-2 kernel
    fl = np.sqrt((l - 1) * (l + 2), where=(l > 0), out=np.zeros(lmax + 1))
    fl /= 2
    if discretized:
        pw0, pw2 = hp.pixwin(nside, lmax=lmax, pol=True)
        fl *= pw2 / pw0
    hp.almxfl(alm, fl, inplace=True)

    # transform to shear maps
    gamma = hp.alm2map_spin([alm, blm], nside, 2, lmax)
    gamma = gamma[0] + 1j * gamma[1]
    results += (gamma,)

    # all done
    return results


def shear_from_convergence(
    kappa: npt.NDArray[np.float64],
    lmax: int | None = None,
    *,
    discretized: bool = True,
) -> npt.NDArray[np.float64]:
    """
    _summary_.

    Parameters
    ----------
    kappa
        _description_
    lmax
        _description_
    discretized
        _description_

    Returns
    -------
        _description_

    """
    nside = hp.get_nside(kappa)
    if lmax is None:
        lmax = 3 * nside - 1

    # compute alm
    alm = hp.map2alm(kappa, lmax=lmax, pol=False, use_pixel_weights=True)

    # zero B-modes
    blm = np.zeros_like(alm)

    # factor to convert convergence alm to shear alm
    l = np.arange(lmax + 1)  # noqa: E741
    fl = np.sqrt((l + 2) * (l + 1) * l * (l - 1))
    fl /= np.clip(l * (l + 1), 1, None)
    fl *= -1

    # if discretised, factor out spin-0 kernel and apply spin-2 kernel
    if discretized:
        pw0, pw2 = hp.pixwin(nside, lmax=lmax, pol=True)
        fl *= pw2 / pw0

    # apply correction to E-modes
    hp.almxfl(alm, fl, inplace=True)

    # transform to shear maps
    return hp.alm2map_spin([alm, blm], nside, 2, lmax)


class MultiPlaneConvergence:
    """_summary_."""

    def __init__(self, cosmo: Cosmology) -> None:
        """
        _summary_.

        Parameters
        ----------
        cosmo
            _description_

        """
        self.cosmo = cosmo

        # set up initial values of variables
        self.z2: float = 0.0
        self.z3: float = 0.0
        self.x3: float = 0.0
        self.w3: float = 0.0
        self.r23: float = 1.0
        self.delta3: npt.NDArray[np.float64] = np.array(0.0)
        self.kappa2: npt.NDArray[np.float64] | None = None
        self.kappa3: npt.NDArray[np.float64] | None = None

    def add_window(self, delta: npt.NDArray[np.float64], w: RadialWindow) -> None:
        """
        _summary_.

        Parameters
        ----------
        delta
            _description_
        w
            _description_

        """
        zsrc = w.zeff
        lens_weight = np.trapz(  # type: ignore[attr-defined]
            w.wa,
            w.za,
        ) / np.interp(
            zsrc,
            w.za,
            w.wa,
        )

        self.add_plane(delta, zsrc, lens_weight)

    def add_plane(
        self,
        delta: npt.NDArray[np.float64],
        zsrc: float,
        wlens: float = 1.0,
    ) -> None:
        """
        _summary_.

        Parameters
        ----------
        delta
            _description_
        zsrc
            _description_
        wlens
            _description_

        Raises
        ------
        ValueError
            _description_

        """
        if zsrc <= self.z3:
            msg = "source redshift must be increasing"
            raise ValueError(msg)

        # cycle mass plane, ...
        delta2, self.delta3 = self.delta3, delta

        # redshifts of source planes, ...
        z1, self.z2, self.z3 = self.z2, self.z3, zsrc

        # and weights of mass plane
        w2, self.w3 = self.w3, wlens

        # extrapolation law
        x2, self.x3 = self.x3, self.cosmo.xm(self.z3)
        r12 = self.r23
        r13, self.r23 = self.cosmo.xm([z1, self.z2], self.z3) / self.x3
        t = r13 / r12

        # lensing weight of mass plane to be added
        f = 3 * self.cosmo.omega_m / 2
        f *= x2 * self.r23
        f *= (1 + self.z2) / self.cosmo.ef(self.z2)
        f *= w2

        # create kappa planes on first iteration
        if self.kappa2 is None:
            self.kappa2 = np.zeros_like(delta)
            self.kappa3 = np.zeros_like(delta)

        # cycle convergence planes
        # normally: kappa1, kappa2, kappa3 = kappa2, kappa3, <empty>
        # but then we set: kappa3 = (1-t)*kappa1 + ...
        # so we can set kappa3 to previous kappa2 and modify in place
        self.kappa2, self.kappa3 = self.kappa3, self.kappa2

        # compute next convergence plane in place of last
        self.kappa3 *= 1 - t
        self.kappa3 += t * self.kappa2
        self.kappa3 += f * delta2

    @property
    def zsrc(self) -> float:
        """
        _summary_.

        Returns
        -------
            _description_

        """
        return self.z3

    @property
    def kappa(self) -> npt.NDArray[np.float64] | None:
        """
        _summary_.

        Returns
        -------
            _description_

        """
        return self.kappa3

    @property
    def delta(self) -> npt.NDArray[np.float64]:
        """
        _summary_.

        Returns
        -------
            _description_

        """
        return self.delta3

    @property
    def wlens(self) -> float:
        """
        _summary_.

        Returns
        -------
            _description_

        """
        return self.w3


def multi_plane_matrix(
    shells: list[RadialWindow],
    cosmo: Cosmology,
) -> npt.NDArray[np.float64]:
    """
    _summary_.

    Parameters
    ----------
    shells
        _description_
    cosmo
        _description_

    Returns
    -------
        _description_

    """
    mpc = MultiPlaneConvergence(cosmo)
    wmat = np.eye(len(shells))
    for i, w in enumerate(shells):
        mpc.add_window(wmat[i].copy(), w)
        wmat[i, :] = mpc.kappa
    return wmat


def multi_plane_weights(
    weights: npt.NDArray[np.float64],
    shells: list[RadialWindow],
    cosmo: Cosmology,
) -> npt.NDArray[np.float64]:
    """
    _summary_.

    Parameters
    ----------
    weights
        _description_
    shells
        _description_
    cosmo
        _description_

    Returns
    -------
        The relative lensing weight of each shell.

    Raises
    ------
    ValueError
        _description_

    """
    # ensure shape of weights ends with the number of shells
    shape = np.shape(weights)
    if not shape or shape[0] != len(shells):
        msg = "shape mismatch between weights and shells"
        raise ValueError(msg)
    # normalise weights
    weights = weights / np.sum(weights, axis=0)
    # combine weights and the matrix of lensing contributions
    mat = multi_plane_matrix(shells, cosmo)
    return np.matmul(mat.T, weights)


def deflect(
    lon: float | npt.NDArray[np.float64],
    lat: float | npt.NDArray[np.float64],
    alpha: complex | list[float] | npt.NDArray[np.complex128] | npt.NDArray[np.float64],
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """
    _summary_.

    Parameters
    ----------
    lon
        _description_
    lat
        _description_
    alpha
        _description_

    Returns
    -------
        The longitudes and latitudes after deflection.

    """
    alpha = np.asanyarray(alpha)
    if np.iscomplexobj(alpha):
        alpha1, alpha2 = alpha.real, alpha.imag
    else:
        alpha1, alpha2 = alpha

    # we know great-circle navigation:
    # θ' = arctan2(√[(cosθ sin|α| - sinθ cos|α| cosγ)² + (sinθ sinγ)²],
    #              cosθ cos|α| + sinθ sin|α| cosγ)
    # δ = arctan2(sin|α| sinγ, sinθ cos|α| - cosθ sin|α| cosγ)

    t = np.radians(lat)
    ct, st = np.sin(t), np.cos(t)  # sin and cos flipped: lat not co-lat

    a = np.hypot(alpha1, alpha2)  # abs(alpha)
    g = np.arctan2(alpha2, alpha1)  # arg(alpha)
    ca, sa = np.cos(a), np.sin(a)
    cg, sg = np.cos(g), np.sin(g)

    # flipped atan2 arguments for lat instead of co-lat
    tp = np.arctan2(ct * ca + st * sa * cg, np.hypot(ct * sa - st * ca * cg, st * sg))

    d = np.arctan2(sa * sg, st * ca - ct * sa * cg)

    return lon - np.degrees(d), np.degrees(tp)
