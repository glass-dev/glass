'''
Galaxy shear
============

This example simulates a galaxy catalogue with shears affected by weak lensing,
combining the :ref:`sphx_glr_examples_1_basic_plot_density.py` and
:ref:`sphx_glr_examples_1_basic_plot_lensing.py` examples with generators for
the intrinsic galaxy ellipticity and the resulting shear.

'''

# %%
# Setup
# -----
# The basic setup of galaxies and weak lensing fields is the same as in the
# previous examples.
#
# In addition, there is a generator for intrinsic galaxy ellipticities,
# following a normal distribution.  The standard deviation is much too small to
# be realistic, but enables the example to get away with fewer total galaxies.
#
# Finally, there is a generator that applies the reduced shear from the lensing
# maps to the intrinsic ellipticities, producing the galaxy shears.

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

# these are the GLASS imports: cosmology, glass modules, and the CAMB module
from cosmology import LCDM
import glass.sim
import glass.camb
import glass.matter
import glass.lensing
import glass.galaxies

# also needs camb itself to get the parameter object, and the expectation
import camb


# cosmology for the simulation
cosmo = LCDM(h=0.7, Om=0.3)

# basic parameters of the simulation
nside = 512
lmax = nside

# the intrinsic galaxy ellipticity
# this is very small so that the galaxy density can be small, too
sigma_e = 0.01

# galaxy density
n_arcmin2 = 0.01

# localised redshift distribution with the given density
z = np.linspace(0, 1, 101)
dndz = np.exp(-(z - 0.5)**2/(0.1)**2)
dndz *= n_arcmin2/np.trapz(dndz, z)

# set up CAMB parameters for matter angular power spectrum
pars = camb.set_params(H0=100*cosmo.h, omch2=cosmo.Om*cosmo.h**2,
                       NonLinear=camb.model.NonLinear_both)

# generators for lensing and galaxies
generators = [
    glass.sim.zspace(0, 1.01, dz=0.1),
    glass.camb.camb_matter_cl(pars, lmax),
    glass.matter.lognormal_matter(nside),
    glass.lensing.convergence(cosmo),
    glass.lensing.shear(),
    glass.galaxies.gal_dist_uniform(z, dndz),
    glass.galaxies.gal_ellip_gaussian(sigma_e),
    glass.galaxies.gal_shear_interp(cosmo),
]


# %%
# Simulation
# ----------
# Simulate the galaxies with shears.  In each iteration, get the shears and map
# them to a HEALPix map for later analysis.

# map for sum of shears
she = np.zeros(hp.nside2npix(nside), dtype=complex)

# keep count of total number of galaxies
num = np.zeros_like(she, dtype=int)

# iterate and map the galaxy shears to a HEALPix map
for it in glass.sim.generate(generators):
    gal_lon, gal_lat = it['gal_lon'], it['gal_lat']
    gal_she = it['gal_she']

    gal_pix = hp.ang2pix(nside, gal_lon, gal_lat, lonlat=True)
    s = np.argsort(gal_pix)
    pix, start, count = np.unique(gal_pix[s], return_index=True, return_counts=True)
    she[pix] += list(map(np.sum, np.split(gal_she[s], start[1:])))
    num[pix] += count


# %%
# Analysis
# --------
# Compute the angular power spectrum of the observed galaxy shears.  To compare
# with the expectation, take into account the expected noise level due to shape
# noise, and the expected mixing matrix for a uniform distribution of points.

# will need number of pixels in map
npix = len(she)

# get the angular power spectra from the galaxy shears
cls = hp.anafast([num, she.real, she.imag], pol=True, lmax=lmax)

# the noise level from discrete observations with shape noise
nl = (4*np.pi/npix)*np.mean(num)*sigma_e**2

# factor transforming convergence to shear
l = np.arange(lmax+1)
fl = (l+2)*(l+1)*l*(l-1)/np.clip(l**2*(l+1)**2, 1, None)

# mixing matrix for uniform distribution of points
b = np.mean(num)/npix/2
a = np.mean(num)**2 - b
mm = a*np.eye(lmax+1, lmax+1)
mm += b*np.arange(1, 2*lmax+2, 2)

# get the expected cls from CAMB
pars.Want_CMB = False
pars.min_l = 1
pars.SourceWindows = [camb.sources.SplinedSourceWindow(z=z, W=dndz, source_type='lensing')]
theory_cls = camb.get_results(pars).get_source_cls_dict(lmax=lmax, raw_cl=True)

# plot the realised and expected cls
plt.plot(l, (2*l+1)*(cls[1] - nl), '-k', lw=2, label='simulation')
plt.plot(l, (2*l+1)*(mm@(fl*theory_cls['W1xW1'])), '-r', lw=2, label='expectation')
plt.xscale('symlog', linthresh=10, linscale=0.5, subs=[2, 3, 4, 5, 6, 7, 8, 9])
plt.yscale('symlog', linthresh=1e-7, linscale=0.5, subs=[2, 3, 4, 5, 6, 7, 8, 9])
plt.xlabel(r'angular mode number $l$')
plt.ylabel(r'angular power spectrum $(2l+1) \, C_l^{EE}$')
plt.legend()
plt.tight_layout()
plt.show()
