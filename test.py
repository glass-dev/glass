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

# make a random visibility map with low NSIDE
# also compute its fsky for the extected galaxy count
v = (np.tanh(hp.synfast(np.exp(-np.arange(lmax+1)), 128)) + 1)/2
fsky = np.mean(v)

generators = [
    # glass.xspace(cosmo, 0., 1.001, dx=100.),
    glass.zspace(0., 1.001, dz=0.1),
    glass.ext.camb.camb_matter_cl(pars, lmax),
    glass.matter.lognormal_matter(nside),
    glass.lensing.convergence(cosmo),
    glass.lensing.shear(lmax),
    glass.lensing.lensing_dist(z, dndz, cosmo),
    glass.observations.vis_constant(v, nside),
    glass.galaxies.gal_dist_fullsky(z, dndz, bz),
    glass.galaxies.gal_ellip_ryden04(-2.2, 1.4, 0.57, 0.21),
    glass.galaxies.gal_shear_interp(cosmo),
]

# print the simulation
for g in generators:
    print(g)

fits = fitsio.FITS('map.fits', 'rw', clobber=True)

plt.ion()
plt.figure()

with glass.logger('debug') as log:

    for shell in glass.generate(generators):

        i, zmin, zmax = shell['#'], shell['zmin'], shell['zmax']

        delta = shell['delta']
        kappa = shell['kappa']
        gamma1 = shell['gamma1']
        gamma2 = shell['gamma2']
        gal_pop = shell['gal_pop']
        gal_z = shell['gal_z']
        gal_g = np.abs(shell['gal_she'])
        gal_lon, gal_lat = shell['gal_lon'], shell['gal_lat']
        vis = shell['visibility']

        assert np.all((zmin <= gal_z) & (gal_z <= zmax))

        # write FITS
        extname = f'MAP{i}'
        header = [
            {'name': 'ZMIN', 'value': zmin, 'comment': 'lower redshift bound'},
            {'name': 'ZMAX', 'value': zmax, 'comment': 'upper redshift bound'},
        ]
        # fits.write_table([delta, kappa], names=['delta', 'kappa'], extname=extname, header=header)

        plt.figure(figsize=(12, 8))

        z_ = np.linspace(zmin, zmax, 100)

        # plot galaxy redshifts
        plt.subplot(2, 2, 2)
        plt.cla()
        plt.title('galaxy redshifts')
        for k in range(2):
            dndz_k = np.interp(z_, z, dndz[k])
            h, _ = np.histogram(gal_z[gal_pop == k], bins=z_)
            zm_ = (z_[:-1]+z_[1:])/2
            plt.plot(zm_, h, '-k')
            plt.plot(zm_, ARCMIN2_SPHERE*fsky*(dndz_k[:-1]+dndz_k[1:])/2*np.diff(z_), ':r')
        plt.xlabel(r'$z$')

        # plot galaxy ellipticities
        plt.subplot(2, 2, 4)
        plt.cla()
        plt.title('galaxy shear')
        plt.hist(gal_g, bins=50, range=[0, 1], histtype='step')
        plt.xlabel(r'$g$')

        # fix layout for plots with axes
        plt.tight_layout()

        # plot maps
        plt.suptitle(f'z = {zmin:.3f} ... {zmax:.3f}')
        plt.subplot(3, 4, 1)
        hp.mollview(delta, title=r'overdensity $\delta$', hold=True)
        plt.subplot(3, 4, 2)
        hp.mollview(kappa, title=r'convergence $\kappa$', hold=True)
        plt.subplot(3, 4, 5)
        hp.mollview(gamma1, title=r'shear component $\gamma_1$', hold=True)
        plt.subplot(3, 4, 6)
        hp.mollview(gamma2, title=r'shear component $\gamma_2$', hold=True)

        # plot galaxies on a low-resolution map
        gals = np.zeros(hp.nside2npix(128))
        ipix = hp.ang2pix(128, gal_lon, gal_lat, lonlat=True)
        ipix, icnt = np.unique(ipix, return_counts=True)
        gals[ipix] = icnt
        plt.subplot(3, 4, 9)
        hp.mollview(vis, title='visibility', hold=True, min=0, max=1)
        plt.subplot(3, 4, 10)
        hp.mollview(gals, title='galaxy counts', hold=True)

        # update plot window
        plt.pause(1e-3)

plt.ioff()

fits.close()
