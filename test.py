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

with glass.logger('debug') as log:

    # zbins = glass.xrange(cosmo, 0., 1.001, dx=100.)
    zbins = glass.zrange(0., 1.001, dz=0.1)

    log.info('zbins: %s', zbins)

    gs = []

    g = glass.cls.cls_from_pyccl(lmax, cosmo)
    gs.append(g)

    g = glass.matter.lognormal_matter(nside)
    gs.append(g)

    g = glass.lensing.convergence_from_matter(cosmo)
    gs.append(g)

    g = glass.plot.interactive_display(['delta', 'kappa'])
    g.inputs(maps=['delta', 'kappa'], points=[])
    gs.append(g)

    g = glass.output.write_map('map.fits', ['delta', 'kappa'], clobber=True)
    g.inputs(maps=['delta', 'kappa'])
    gs.append(g)

    glass.simulate(zbins, gs)
