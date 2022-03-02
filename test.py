from cosmology import LCDM

import numpy as np
import healpy as hp
import fitsio
import matplotlib.pyplot as plt

import glass

import glass.ext.camb
import camb


ARCMIN2_SPHERE = 60**6//100/np.pi


cosmo = LCDM(h=0.7, Om=0.3)

nside = 1024
lmax = nside

# galaxy density
n_arcmin2 = .3

# source distribution
z = np.linspace(0, 2, 1000)
dndz = np.exp(-0.5*(z - [[0.3], [0.8]])**2/(0.1)**2)
dndz /= np.trapz(dndz, z, axis=-1)[..., np.newaxis]
dndz *= n_arcmin2
bz = 0.8

# set up CAMB for matter cls
pars = camb.set_params(H0=100*cosmo.h, omch2=cosmo.Om*cosmo.h**2)

generators = [
    # glass.xspace(cosmo, 0., 1.001, dx=100.),
    glass.zspace(0., 1.001, dz=0.1),
    glass.ext.camb.camb_matter_cl(pars, lmax),
    glass.matter.lognormal_matter(nside),
    glass.lensing.convergence(cosmo),
    glass.lensing.shear(lmax),
    glass.galaxies.galdist_fullsky(z, dndz, bz),
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
        gal_pop = shell['gal_pop']
        gal_z = shell['gal_z']

        assert np.all((zmin <= gal_z) & (gal_z <= zmax))

        # write FITS
        extname = f'MAP{i}'
        header = [
            {'name': 'ZMIN', 'value': zmin, 'comment': 'lower redshift bound'},
            {'name': 'ZMAX', 'value': zmax, 'comment': 'upper redshift bound'},
        ]
        fits.write_table([delta, kappa], names=['delta', 'kappa'], extname=extname, header=header)

        # plot maps
        plt.suptitle(f'z = {zmin:.3f} ... {zmax:.3f}')
        plt.subplot(2, 3, 1)
        hp.mollview(delta, title=r'overdensity $\delta$', hold=True)
        plt.subplot(2, 3, 2)
        hp.mollview(kappa, title=r'convergence $\kappa$', hold=True)
        plt.subplot(2, 3, 3)
        hp.mollview(gamma1, title=r'shear component $\gamma_1$', hold=True)
        plt.subplot(2, 3, 4)
        hp.mollview(gamma2, title=r'shear component $\gamma_2$', hold=True)

        z_ = np.linspace(zmin, zmax, 100)

        # plot galaxies
        for k in range(2):
            dndz_k = np.interp(z_, z, dndz[k])
            plt.subplot(2, 3, 5+k)
            plt.cla()
            plt.hist(gal_z[gal_pop == k], bins=z_, density=True, histtype='stepfilled', alpha=0.3)
            plt.plot(z_, dndz_k/np.trapz(dndz_k, z_), ':')

        # update plot window
        plt.pause(1e-3)

plt.ioff()

fits.close()
