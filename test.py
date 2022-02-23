from cosmology import LCDM

import healpy as hp
import fitsio
import matplotlib.pyplot as plt

import glass

import glass.ext.camb
import camb


cosmo = LCDM(h=0.7, Om=0.3)

nside = 1024
lmax = nside


# set up CAMB for matter cls
pars = camb.set_params(H0=100*cosmo.h, omch2=cosmo.Om*cosmo.h**2)


generators = [
    # glass.xspace(cosmo, 0., 1.001, dx=100.),
    glass.zspace(0., 1.001, dz=0.1),
    glass.ext.camb.camb_matter_cl(pars, lmax),
    glass.matter.lognormal_matter(nside),
    glass.lensing.multiplane_convergence(cosmo),
    glass.lensing.shear(lmax),
]

# print the simulation
for g in generators:
    print(g)

fits = fitsio.FITS('map.fits', 'rw', clobber=True)

plt.ion()
plt.figure()

with glass.logger('debug') as log:

    for shell in glass.lightcone(generators):

        i, zmin, zmax = shell['#'], shell['zmin'], shell['zmax']

        delta = shell['delta']
        kappa = shell['kappa']
        gamma1 = shell['gamma1']
        gamma2 = shell['gamma2']

        # write FITS
        extname = f'MAP{i}'
        header = [
            {'name': 'ZMIN', 'value': zmin, 'comment': 'lower redshift bound'},
            {'name': 'ZMAX', 'value': zmax, 'comment': 'upper redshift bound'},
        ]
        fits.write_table([delta, kappa], names=['delta', 'kappa'], extname=extname, header=header)

        # plot maps
        plt.suptitle(f'z = {zmin:.3f} ... {zmax:.3f}')
        plt.subplot(2, 2, 1)
        hp.mollview(delta, title=r'overdensity $\delta$', hold=True)
        plt.subplot(2, 2, 2)
        hp.mollview(kappa, title=r'convergence $\kappa$', hold=True)
        plt.subplot(2, 2, 3)
        hp.mollview(gamma1, title=r'shear component $\gamma_1$', hold=True)
        plt.subplot(2, 2, 4)
        hp.mollview(gamma2, title=r'shear component $\gamma_2$', hold=True)
        plt.pause(1e-3)

plt.ioff()

fits.close()
