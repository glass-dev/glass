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

"""  # noqa: D400

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

import healpy as hp
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    import glass
    from glass.cosmology import Cosmology


@overload
def from_convergence(
    kappa: NDArray[np.float64],
    lmax: int | None = None,
    *,
    potential: Literal[True] = True,
    deflection: Literal[False] = False,
    shear: Literal[False] = False,
    discretized: bool = True,
) -> tuple[NDArray[np.float64]]:
    # returns psi
    ...


@overload
def from_convergence(
    kappa: NDArray[np.float64],
    lmax: int | None = None,
    *,
    potential: Literal[False] = False,
    deflection: Literal[True] = True,
    shear: Literal[False] = False,
    discretized: bool = True,
) -> tuple[NDArray[np.complex128]]:
    # returns alpha
    ...


@overload
def from_convergence(
    kappa: NDArray[np.float64],
    lmax: int | None = None,
    *,
    potential: Literal[False] = False,
    deflection: Literal[False] = False,
    shear: Literal[True] = True,
    discretized: bool = True,
) -> tuple[NDArray[np.complex128]]:
    # returns gamma
    ...


@overload
def from_convergence(
    kappa: NDArray[np.float64],
    lmax: int | None = None,
    *,
    potential: Literal[True] = True,
    deflection: Literal[True] = True,
    shear: Literal[False] = False,
    discretized: bool = True,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.complex128],
]:
    # returns psi, alpha
    ...


@overload
def from_convergence(
    kappa: NDArray[np.float64],
    lmax: int | None = None,
    *,
    potential: Literal[True] = True,
    deflection: Literal[False] = False,
    shear: Literal[True] = True,
    discretized: bool = True,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.complex128],
]:
    # returns psi, gamma
    ...


@overload
def from_convergence(
    kappa: NDArray[np.float64],
    lmax: int | None = None,
    *,
    potential: Literal[False] = False,
    deflection: Literal[True] = True,
    shear: Literal[True] = True,
    discretized: bool = True,
) -> tuple[
    NDArray[np.complex128],
    NDArray[np.complex128],
]:
    # returns alpha, gamma
    ...


@overload
def from_convergence(
    kappa: NDArray[np.float64],
    lmax: int | None = None,
    *,
    potential: Literal[True] = True,
    deflection: Literal[True] = True,
    shear: Literal[True] = True,
    discretized: bool = True,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.complex128],
    NDArray[np.complex128],
]:
    # returns psi, alpha, gamma
    ...


def from_convergence(  # noqa: PLR0913
    kappa: NDArray[np.float64],
    lmax: int | None = None,
    *,
    potential: bool = False,
    deflection: bool = False,
    shear: bool = False,
    discretized: bool = True,
) -> tuple[NDArray[np.float64] | NDArray[np.complex128], ...]:
    r"""
    Compute other weak lensing maps from the convergence.

    Takes a weak lensing convergence map and returns one or more of
    deflection potential, deflection, and shear maps. The maps are
    computed via spherical harmonic transforms.

    Parameters
    ----------
    kappa
        HEALPix map of the convergence field.
    lmax
        Maximum angular mode number to use in the transform.
    potential
        Which lensing maps to return.
    deflection
        Which lensing maps to return.
    shear
        Which lensing maps to return.
    discretized
        Correct the pixel window function in output maps.

    Returns
    -------
    psi
        Map of the lensing (or deflection) potential. Only returned if
        ``potential`` is true.
    alpha
        Map of the deflection (complex). Only returned if ``deflection`` if true.
    gamma
        Map of the shear (complex). Only returned if ``shear`` is true.

    Notes
    -----
    The weak lensing fields are computed from the convergence or
    deflection potential in the following way. [Tessore23]_

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
    ell = np.arange(lmax + 1)

    # this tuple will be returned
    results: tuple[NDArray[np.float64] | NDArray[np.complex128], ...] = ()

    # convert convergence to potential
    fl = np.divide(-2, ell * (ell + 1), where=(ell > 0), out=np.zeros(lmax + 1))
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
    fl = np.sqrt(ell * (ell + 1))
    # missing spin-1 pixel window function here
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
    fl = np.sqrt((ell - 1) * (ell + 2), where=(ell > 0), out=np.zeros(lmax + 1))
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
    kappa: NDArray[np.float64],
    lmax: int | None = None,
    *,
    discretized: bool = True,
) -> NDArray[np.float64]:
    """
    Weak lensing shear from convergence.

    Computes the shear from the convergence using a spherical harmonic
    transform.

    .. deprecated:: 2023.6
       Use the more general :func:`glass.from_convergence` function instead.

    Parameters
    ----------
    kappa
        The convergence map.
    lmax
        The maximum angular mode number to use in the transform.
    discretized
        Whether to correct the pixel window function in the output map.

    Returns
    -------
        The shear map.

    """
    nside = hp.get_nside(kappa)
    if lmax is None:
        lmax = 3 * nside - 1

    # compute alm
    alm = hp.map2alm(kappa, lmax=lmax, pol=False, use_pixel_weights=True)

    # zero B-modes
    blm = np.zeros_like(alm)

    # factor to convert convergence alm to shear alm
    ell = np.arange(lmax + 1)
    fl = np.sqrt((ell + 2) * (ell + 1) * ell * (ell - 1))
    fl /= np.clip(ell * (ell + 1), 1, None)
    fl *= -1

    # if discretised, factor out spin-0 kernel and apply spin-2 kernel
    if discretized:
        pw0, pw2 = hp.pixwin(nside, lmax=lmax, pol=True)
        fl *= pw2 / pw0

    # apply correction to E-modes
    hp.almxfl(alm, fl, inplace=True)

    # transform to shear maps
    return hp.alm2map_spin([alm, blm], nside, 2, lmax)  # type: ignore[no-any-return]


class MultiPlaneConvergence:
    """Compute convergence fields iteratively from multiple matter planes."""

    def __init__(self, cosmo: Cosmology) -> None:
        """
        Create a new instance to iteratively compute the convergence.

        Parameters
        ----------
        cosmo
            Cosmology instance.

        """
        self.cosmo = cosmo

        # set up initial values of variables
        self.z2: float | NDArray[np.float64] = 0.0
        self.z3: float | NDArray[np.float64] = 0.0
        self.x3: float = 0.0
        self.w3: float = 0.0
        self.r23: float = 1.0
        self.delta3: NDArray[np.float64] = np.array(0.0)
        self.kappa2: NDArray[np.float64] | None = None
        self.kappa3: NDArray[np.float64] | None = None

    def add_window(self, delta: NDArray[np.float64], w: glass.RadialWindow) -> None:
        """
        Add a mass plane from a window function to the convergence.

        The lensing weight is computed from the window function, and the
        source plane redshift is the effective redshift of the window.

        Parameters
        ----------
        delta
            The mass plane.
        w
            The window function.

        """
        zsrc = w.zeff
        lens_weight = float(np.trapezoid(w.wa, w.za) / np.interp(zsrc, w.za, w.wa))

        self.add_plane(delta, zsrc, lens_weight)

    def add_plane(
        self,
        delta: NDArray[np.float64],
        zsrc: float | NDArray[np.float64],
        wlens: float = 1.0,
    ) -> None:
        """
        Add a mass plane at redshift ``zsrc`` to the convergence.

        Parameters
        ----------
        delta
            The mass plane.
        zsrc
            The redshift of the source plane.
        wlens
            The weight of the mass plane.

        Raises
        ------
        ValueError
            If the source redshift is not increasing.

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
        x2, self.x3 = (
            self.x3,
            self.cosmo.transverse_comoving_distance(self.z3)
            / self.cosmo.hubble_distance,
        )
        r12 = self.r23
        r13, self.r23 = self.cosmo.transverse_comoving_distance(
            [z1, self.z2], self.z3
        ) / (self.cosmo.hubble_distance * self.x3)
        t = r13 / r12

        # lensing weight of mass plane to be added
        f = 3 * self.cosmo.Omega_m0 / 2
        f *= x2 * self.r23
        f *= (1 + self.z2) / self.cosmo.H_over_H0(self.z2)
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
    def zsrc(self) -> float | NDArray[np.float64]:
        """The redshift of the current convergence plane."""
        return self.z3

    @property
    def kappa(self) -> NDArray[np.float64] | None:
        """The current convergence plane."""
        return self.kappa3

    @property
    def delta(self) -> NDArray[np.float64]:
        """The current matter plane."""
        return self.delta3

    @property
    def wlens(self) -> float:
        """The weight of the current matter plane."""
        return self.w3


def multi_plane_matrix(
    shells: Sequence[glass.RadialWindow],
    cosmo: Cosmology,
) -> NDArray[np.float64]:
    """
    Compute the matrix of lensing contributions from each shell.

    Parameters
    ----------
    shells
        The shells of the mass distribution.
    cosmo
        Cosmology instance.

    Returns
    -------
        The matrix of lensing contributions.

    """
    mpc = MultiPlaneConvergence(cosmo)
    wmat = np.eye(len(shells))
    for i, w in enumerate(shells):
        mpc.add_window(wmat[i].copy(), w)
        wmat[i, :] = mpc.kappa
    return wmat


def multi_plane_weights(
    weights: NDArray[np.float64],
    shells: Sequence[glass.RadialWindow],
    cosmo: Cosmology,
) -> NDArray[np.float64]:
    """
    Compute effective weights for multi-plane convergence.

    Converts an array *weights* of relative weights for each shell
    into the equivalent array of relative lensing weights.

    This is the discretised version of the integral that turns a
    redshift distribution :math:`n(z)` into the lensing efficiency
    sometimes denoted :math:`g(z)` or :math:`q(z)`.

    Parameters
    ----------
    weights
        Relative weight of each shell. The first axis must broadcast
        against the number of shells, and is normalised internally.
    shells
        Window functions of the shells.
    cosmo
        Cosmology instance.

    Returns
    -------
        The relative lensing weight of each shell.

    Raises
    ------
    ValueError
        If the shape of *weights* does not match the number of shells.

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
    return np.matmul(mat.T, weights)  # type: ignore[no-any-return]


def deflect(
    lon: float | NDArray[np.float64],
    lat: float | NDArray[np.float64],
    alpha: complex | list[float] | NDArray[np.complex128] | NDArray[np.float64],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
]:
    r"""
    Apply deflections to positions.

    Takes an array of :term:`deflection` values and applies them
    to the given positions.

    Parameters
    ----------
    lon
        Longitudes to be deflected.
    lat
        Latitudes to be deflected.
    alpha
        Deflection values. Must be complex-valued or have a leading
        axis of size 2 for the real and imaginary component.

    Returns
    -------
        The longitudes and latitudes after deflection.

    Notes
    -----
    Deflections on the sphere are :term:`defined <deflection>` as
    follows:  The complex deflection :math:`\alpha` transports a point
    on the sphere an angular distance :math:`|\alpha|` along the
    geodesic with bearing :math:`\arg\alpha` in the original point.

    In the language of differential geometry, this function is the
    exponential map.

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
