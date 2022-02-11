import logging
import numpy as np

from cosmology import LCDM

from glass.cls import cls_from_pyccl
from glass.random import lognormal_fields
from glass.lensing import convergence_from_matter
from glass.analysis import write_map
from glass.plotting import interactive_display
from glass.simulation import simulate


log = logging.getLogger('glass')
log.setLevel(logging.DEBUG)

log_h = logging.StreamHandler()
log_h.setLevel(logging.DEBUG)
log_f = logging.Formatter('%(message)s')
log_h.setFormatter(log_f)
log.addHandler(log_h)


cosmo = LCDM(H0=70, Om=0.3)

dx = 330.
nbins = 10
zbins = cosmo.dc_inv(dx*np.arange(nbins+1))

print('zbins:', zbins)

nside = 1024
lmax = 3*nside-1

generators = [
    (cls_from_pyccl(lmax, cosmo), ('zmin', 'zmax'), 'cls'),
    (lognormal_fields(nside), 'cls', 'delta'),
    (convergence_from_matter(cosmo), ('zmin', 'zmax', 'delta'), 'kappa'),
    (interactive_display(['delta', 'kappa'], []), ('zmin', 'zmax', ('delta', 'kappa'), ()), None),
    (write_map('map.fits', names=['delta', 'kappa'], clobber=True), ('zmin', 'zmax', 'delta', 'kappa'), None),
]

simulate(zbins, generators)
