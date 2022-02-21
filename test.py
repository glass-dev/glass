from cosmology import LCDM

import healpy as hp
import fitsio
import matplotlib.pyplot as plt

import glass
import glass.cls


cosmo = LCDM(h=0.7, Om=0.3)

nside = 1024
lmax = nside

generators = [
    # glass.xspace(cosmo, 0., 1.001, dx=100.),
    glass.zspace(0., 1.001, dz=0.1),
    glass.cls.cls_from_pyccl(lmax, cosmo),
    glass.matter.lognormal_matter(nside),
    glass.lensing.convergence_from_matter(cosmo),
]

# print the simulation
for g in generators:
    print(g)

fits = fitsio.FITS('maps.fits', 'rw', clobber=True)

plt.ion()
plt.figure()

with glass.logger('debug') as log:

    for shell in glass.lightcone(generators):

        i, zmin, zmax = shell['#'], shell['zmin'], shell['zmax']

        delta = shell['delta']
        kappa = shell['kappa']

        # write FITS
        extname = f'MAP{i}'
        header = [
            {'name': 'ZMIN', 'value': zmin, 'comment': 'lower redshift bound'},
            {'name': 'ZMAX', 'value': zmax, 'comment': 'upper redshift bound'},
        ]
        fits.write_table([delta, kappa], names=['delta', 'kappa'], extname=extname, header=header)

        # plot maps
        plt.suptitle(f'z = {zmin:.3f} ... {zmax:.3f}')
        plt.subplot(1, 2, 1)
        hp.mollview(delta, title=r'overdensity $\delta$', hold=True)
        plt.subplot(1, 2, 2)
        hp.mollview(kappa, title=r'convergence $\kappa$', hold=True)
        plt.pause(1e-3)

plt.ioff()

fits.close()
