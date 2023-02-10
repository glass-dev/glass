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
.. autofunction:: multi_plane_weights
.. autofunction:: multi_plane_matrix


Lensing fields
--------------

.. autofunction:: shear_from_convergence

'''

from typing import Optional, Sequence, TYPE_CHECKING
import numpy as np
import healpy as hp

if TYPE_CHECKING:
    from cosmology import Cosmology
    from .matter import MatterWeights


def shear_from_convergence(kappa: np.ndarray, lmax: Optional[int] = None, *,
                           discretized: bool = True) -> np.ndarray:
    r'''weak lensing shear from convergence

    Notes
    -----
    The shear field is computed from the convergence or deflection potential in
    the following way.

    Define the spin-raising and spin-lowering operators of the spin-weighted
    spherical harmonics as

    .. math::

        \eth {}_sY_{lm}
        = +\sqrt{(l-s)(l+s+1)} \, {}_{s+1}Y_{lm} \;, \\
        \bar{\eth} {}_sY_{lm}
        = -\sqrt{(l+s)(l-s+1)} \, {}_{s-1}Y_{lm} \;.

    The convergence field :math:`\kappa` is related to the deflection potential
    field :math:`\phi` as

    .. math::

        2 \kappa = \eth\bar{\eth} \, \phi = \bar{\eth}\eth \, \phi \;.

    The convergence modes :math:`\kappa_{lm}` are hence related to the
    deflection potential modes :math:`\phi_{lm}` as

    .. math::

        2 \kappa_{lm} = -l \, (l+1) \, \phi_{lm} \;.

    The shear field :math:`\gamma` is related to the deflection potential field
    as

    .. math::

        2 \gamma = \eth\eth \, \phi
        \quad\text{or}\quad
        2 \gamma = \bar{\eth}\bar{\eth} \, \phi \;,

    depending on the definition of the shear field spin weight as :math:`2` or
    :math:`-2`.  In either case, the shear modes :math:`\gamma_{lm}` are
    related to the deflection potential modes as

    .. math::

        2 \gamma_{lm} = \sqrt{(l+2) \, (l+1) \, l \, (l-1)} \, \phi_{lm} \;.

    The shear modes can therefore be obtained via the convergence, or
    directly from the deflection potential.

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


def multi_plane_weights(zsrc: Sequence[float], weights: 'MatterWeights'
                        ) -> np.ndarray:
    '''Compute weights for multi-plane lensing from matter weights.'''
    return np.array([np.trapz(w, z)/np.interp(z_, z, w)
                     for z_, z, w in zip(zsrc, weights.z, weights.w)])


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

    def add_plane(self, delta: np.ndarray, z: float, w: float = 1.) -> None:
        '''Add a mass plane at redshift ``z`` to the convergence.'''

        if z <= self.z3:
            raise ValueError('source redshift must be increasing')

        # cycle mass plane, ...
        delta2, self.delta3 = self.delta3, delta

        # redshifts of source planes, ...
        z1, self.z2, self.z3 = self.z2, self.z3, z

        # and weights of mass plane
        w2, self.w3 = self.w3, w

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
    def z(self) -> float:
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
    def w(self) -> float:
        '''The weight of the current matter plane.'''
        return self.w3


def multi_plane_matrix(redshifts: Sequence[float], weights: Sequence[float],
                       cosmo: 'Cosmology') -> np.ndarray:
    '''Compute the matrix of lensing contribution from each shell.'''
    mpc = MultiPlaneConvergence(cosmo)
    mat = np.eye(len(redshifts))
    for m, z, w in zip(mat, redshifts, weights):
        mpc.add_plane(m.copy(), z, w)
        m[:] = mpc.kappa
    return mat
