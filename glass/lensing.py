# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''
Lensing (:mod:`glass.lensing`)
==============================

.. currentmodule:: glass.lensing

The :mod:`glass.lensing` module provides functionality for simulating
gravitational lensing by the matter distribution in the universe.

Iterative lensing
-----------------

.. autoclass:: MultiPlaneConvergence
.. autofunction:: multi_plane_matrix


Lensing fields
--------------

.. autofunction:: from_convergence
.. autofunction:: shear_from_convergence


Applying lensing
----------------

.. autofunction:: deflect

'''

import numpy as np
import healpy as hp

# typing support
from typing import Optional, Sequence, Tuple, TYPE_CHECKING
from numpy.typing import NDArray, ArrayLike
if TYPE_CHECKING:
    # to prevent circular dependencies, only import these for type checking
    from cosmology import Cosmology
    from .shells import RadialWindow


def from_convergence(kappa: NDArray, lmax: Optional[int] = None, *,
                     potential: bool = False,
                     deflection: bool = False,
                     shear: bool = False,
                     discretized: bool = True,
                     ) -> Tuple[NDArray, ...]:
    r'''Compute other weak lensing maps from the convergence.

    Takes a weak lensing convergence map and returns one or more of
    deflection potential, deflection, and shear maps.  The maps are
    computed via spherical harmonic transforms.

    Parameters
    ----------
    kappa : array_like
        HEALPix map of the convergence field.
    lmax : int, optional
        Maximum angular mode number to use in the transform.
    potential, deflection, shear : bool, optional
        Which lensing maps to return.

    Returns
    -------
    psi : array_like
        Map of the deflection potential.  Only returned if ``potential``
        is true.
    alpha : array_like
        Map of the deflection (complex).  Only returned if ``deflection``
        if true.
    gamma : array_like
        Map of the shear (complex).  Only returned if ``shear`` is true.

    Notes
    -----
    The weak lensing fields are computed from the convergence or
    deflection potential in the following way. [1]_

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
    deflection potential :math:`\psi`.  On the sphere, this is

    .. math::

        \alpha
        = \eth \, \psi \;.

    The deflection field has spin weight :math:`1` in the HEALPix
    convention, in order for points to be deflected towards regions of
    positive convergence.  The modes :math:`\alpha_{lm}` of the
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

    and thus has spin weight :math:`2`.  The shear modes
    :math:`\gamma_{lm}` are related to the deflection potential modes as

    .. math::

        2 \gamma_{lm}
        = \sqrt{(l+2) \, (l+1) \, l \, (l-1)} \, \psi_{lm} \;.

    References
    ----------
    .. [1] Tessore N., et al., OJAp, 6, 11 (2023).
           doi:10.21105/astro.2302.01942

    '''

    # no output means no computation, return empty tuple
    if not (potential or deflection or shear):
        return ()

    # get the NSIDE parameter
    nside = hp.get_nside(kappa)
    if lmax is None:
        lmax = 3*nside - 1

    # compute alm
    alm = hp.map2alm(kappa, lmax=lmax, pol=False, use_pixel_weights=True)

    # mode number; all conversions are factors of this
    l = np.arange(lmax+1)

    # this tuple will be returned
    results = ()

    # convert convergence to potential
    fl = np.divide(-2, l*(l+1), where=(l > 0), out=np.zeros(lmax+1))
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
    fl = np.sqrt(l*(l+1))
    # TODO: missing spin-1 pixel window function here
    hp.almxfl(alm, fl, inplace=True)

    # if deflection is requested, compute spin-1 maps and add to output
    if deflection:
        alpha = hp.alm2map_spin([alm, blm], nside, 1, lmax)
        alpha = alpha[0] + 1j*alpha[1]
        results += (alpha,)

    # if no shear is requested, stop here
    if not shear:
        return results

    # compute shear alms in place
    # if discretised, factor out spin-0 kernel and apply spin-2 kernel
    fl = np.sqrt((l-1)*(l+2), where=(l > 0), out=np.zeros(lmax+1))
    fl /= 2
    if discretized:
        pw0, pw2 = hp.pixwin(nside, lmax=lmax, pol=True)
        fl *= pw2/pw0
    hp.almxfl(alm, fl, inplace=True)

    # transform to shear maps
    gamma = hp.alm2map_spin([alm, blm], nside, 2, lmax)
    gamma = gamma[0] + 1j*gamma[1]
    results += (gamma,)

    # all done
    return results


def shear_from_convergence(kappa: np.ndarray, lmax: Optional[int] = None, *,
                           discretized: bool = True) -> np.ndarray:
    r'''Weak lensing shear from convergence.

    .. deprecated:: 2023.6
       Use the more general :func:`from_convergence` function instead.

    Computes the shear from the convergence using a spherical harmonic
    transform.

    '''

    nside = hp.get_nside(kappa)
    if lmax is None:
        lmax = 3*nside - 1

    # compute alm
    alm = hp.map2alm(kappa, lmax=lmax, pol=False, use_pixel_weights=True)

    # zero B-modes
    blm = np.zeros_like(alm)

    # factor to convert convergence alm to shear alm
    l = np.arange(lmax+1)
    fl = np.sqrt((l+2)*(l+1)*l*(l-1))
    fl /= np.clip(l*(l+1), 1, None)
    fl *= -1

    # if discretised, factor out spin-0 kernel and apply spin-2 kernel
    if discretized:
        pw0, pw2 = hp.pixwin(nside, lmax=lmax, pol=True)
        fl *= pw2/pw0

    # apply correction to E-modes
    hp.almxfl(alm, fl, inplace=True)

    # transform to shear maps
    return hp.alm2map_spin([alm, blm], nside, 2, lmax)


class MultiPlaneConvergence:
    '''Compute convergence fields iteratively from multiple matter planes.'''

    def __init__(self, cosmo: 'Cosmology') -> None:
        '''Create a new instance to iteratively compute the convergence.'''
        self.cosmo = cosmo

        # set up initial values of variables
        self.z2: float = 0.
        self.z3: float = 0.
        self.x3: float = 0.
        self.w3: float = 0.
        self.r23: float = 1.
        self.delta3: np.ndarray = np.array(0.)
        self.kappa2: Optional[np.ndarray] = None
        self.kappa3: Optional[np.ndarray] = None

    def add_window(self, delta: np.ndarray, w: 'RadialWindow') -> None:
        '''Add a mass plane from a window function to the convergence.

        The lensing weight is computed from the window function, and the
        source plane redshift is the effective redshift of the window.

        '''

        zsrc = w.zeff
        lens_weight = np.trapz(w.wa, w.za)/np.interp(zsrc, w.za, w.wa)

        self.add_plane(delta, zsrc, lens_weight)

    def add_plane(self, delta: np.ndarray, zsrc: float, wlens: float = 1.
                  ) -> None:
        '''Add a mass plane at redshift ``zsrc`` to the convergence.'''

        if zsrc <= self.z3:
            raise ValueError('source redshift must be increasing')

        # cycle mass plane, ...
        delta2, self.delta3 = self.delta3, delta

        # redshifts of source planes, ...
        z1, self.z2, self.z3 = self.z2, self.z3, zsrc

        # and weights of mass plane
        w2, self.w3 = self.w3, wlens

        # extrapolation law
        x2, self.x3 = self.x3, self.cosmo.xm(self.z3)
        r12 = self.r23
        r13, self.r23 = self.cosmo.xm([z1, self.z2], self.z3)/self.x3
        t = r13/r12

        # lensing weight of mass plane to be added
        f = 3*self.cosmo.omega_m/2
        f *= x2*self.r23
        f *= (1 + self.z2)/self.cosmo.ef(self.z2)
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
        self.kappa3 += t*self.kappa2
        self.kappa3 += f*delta2

    @property
    def zsrc(self) -> float:
        '''The redshift of the current convergence plane.'''
        return self.z3

    @property
    def kappa(self) -> Optional[np.ndarray]:
        '''The current convergence plane.'''
        return self.kappa3

    @property
    def delta(self) -> np.ndarray:
        '''The current matter plane.'''
        return self.delta3

    @property
    def wlens(self) -> float:
        '''The weight of the current matter plane.'''
        return self.w3


def multi_plane_matrix(ws: Sequence['RadialWindow'], cosmo: 'Cosmology'
                       ) -> np.ndarray:
    '''Compute the matrix of lensing contributions from each shell.'''
    mpc = MultiPlaneConvergence(cosmo)
    wmat = np.eye(len(ws))
    for i, w in enumerate(ws):
        mpc.add_window(wmat[i].copy(), w)
        wmat[i, :] = mpc.kappa
    return wmat


def deflect(lon: ArrayLike, lat: ArrayLike, alpha: ArrayLike) -> NDArray:
    """Apply deflections to positions.

    Takes an array of :term:`deflection` values and applies them
    to the given positions.

    Parameters
    ----------
    lon, lat : array_like
        Longitudes and latitudes to be deflected.
    alpha : array_like
        Deflection values.  Must be complex-valued or have a leading
        axis of size 2 for the real and imaginary component.

    Returns
    -------
    lon, lat : array_like
        Longitudes and latitudes after deflection.

    Notes
    -----
    Deflections on the sphere are :term:`defined <deflection>` as
    follows:  The complex deflection :math:`\\alpha` transports a point
    on the sphere an angular distance :math:`|\\alpha|` along the
    geodesic with bearing :math:`\\arg\\alpha` in the original point.

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
    ct, st = np.sin(t), np.cos(t)   # sin and cos flipped: lat not co-lat

    a = np.hypot(alpha1, alpha2)    # abs(alpha)
    g = np.arctan2(alpha2, alpha1)  # arg(alpha)
    ca, sa = np.cos(a), np.sin(a)
    cg, sg = np.cos(g), np.sin(g)

    # flipped atan2 arguments for lat instead of co-lat
    tp = np.arctan2(ct*ca + st*sa*cg, np.hypot(ct*sa - st*ca*cg, st*sg))

    d = np.arctan2(sa*sg, st*ca - ct*sa*cg)

    return lon - np.degrees(d), np.degrees(tp)
