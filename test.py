import logging
import numpy as np

from cosmology import LCDM

from glass.cls import cls_from_pyccl
from glass.random import normal_fields, lognormal_fields
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


cosmo = LCDM(h=0.7, Om=0.3)

dz = 0.1
zbins = np.arange(0., 1.001, dz)

print('zbins:', zbins)

nside = 1024
lmax = nside

generators = [
    (cls_from_pyccl(lmax, cosmo), ('zmin', 'zmax'), 'cls'),
    (lognormal_fields(nside), 'cls', 'delta'),
    (convergence_from_matter(cosmo), ('zmin', 'zmax', 'delta'), 'kappa'),
    (interactive_display(['delta', 'kappa'], []), ('zmin', 'zmax', ('delta', 'kappa'), ()), None),
    (write_map('map.fits', names=['delta', 'kappa'], clobber=True), ('zmin', 'zmax', 'delta', 'kappa'), None),
]

simulate(zbins, generators)
