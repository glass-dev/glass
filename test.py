from cosmology import LCDM

import glass
import glass.cls
import glass.matter
import glass.lensing
import glass.plot
import glass.output


cosmo = LCDM(h=0.7, Om=0.3)

nside = 1024
lmax = nside

generators = []

# g = glass.xspace(cosmo, 0., 1.001, dx=100.)
g = glass.zspace(0., 1.001, dz=0.1)
generators.append(g)

g = glass.cls.cls_from_pyccl(lmax, cosmo)
generators.append(g)

g = glass.matter.lognormal_matter(nside)
generators.append(g)

g = glass.lensing.convergence_from_matter(cosmo)
generators.append(g)

g = glass.plot.interactive_display(['delta', 'kappa'])
g.inputs(maps=['delta', 'kappa'], points=[])
generators.append(g)

g = glass.output.write_map('map.fits', ['delta', 'kappa'], clobber=True)
g.inputs(maps=['delta', 'kappa'])
generators.append(g)

with glass.logger('debug') as log:
    for shell in glass.lightcone(generators):
        pass
