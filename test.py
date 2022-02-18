import logging
import numpy as np

from cosmology import LCDM

from glass.cls import cls_from_pyccl
from glass.matter import lognormal_matter
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

generators = []

g = cls_from_pyccl(lmax, cosmo)
generators.append(g)

g = lognormal_matter(nside)
generators.append(g)

g = convergence_from_matter(cosmo)
generators.append(g)

g = interactive_display(['delta', 'kappa'])
g.inputs(maps=['delta', 'kappa'], points=[])
generators.append(g)

g = write_map('map.fits', ['delta', 'kappa'], clobber=True)
g.inputs(maps=['delta', 'kappa'])
generators.append(g)

simulate(zbins, generators)
