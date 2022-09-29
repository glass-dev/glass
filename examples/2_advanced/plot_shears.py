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

# these are the GLASS imports: cosmology and the glass meta-module
from cosmology import LCDM
from glass import glass

# also needs camb itself to get the parameter object, and the expectation
import camb


# cosmology for the simulation
h = 0.7
Oc = 0.25
Ob = 0.05
cosmo = LCDM(h=h, Om=Oc+Ob)

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
pars = camb.set_params(H0=100*h, omch2=Oc*h**2, ombh2=Ob*h**2)

# generators for lensing and galaxies
generators = [
    glass.sim.zspace(0, 1., dz=0.1),
    glass.matter.mat_wht_redshift(),
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

# get the angular power spectra from the galaxy shears
cls = hp.anafast([num, she.real, she.imag], pol=True, lmax=lmax, use_pixel_weights=True)

# get the theory cls from CAMB
pars.NonLinear = 'NonLinear_both'
pars.Want_CMB = False
pars.min_l = 1
pars.set_for_lmax(lmax)
pars.SourceWindows = [camb.sources.SplinedSourceWindow(z=z, W=dndz, source_type='lensing')]
theory_cls = camb.get_results(pars).get_source_cls_dict(lmax=lmax, raw_cl=True)

# factor transforming convergence to shear
l = np.arange(lmax+1)
fl = (l+2)*(l+1)*l*(l-1)/np.clip(l**2*(l+1)**2, 1, None)

# number of arcmin2 in sphere
ARCMIN2_SPHERE = 60**6//100/np.pi

# will need number of pixels in map for the expectation
npix = len(she)

# compute the mean number of shears per pixel
nbar = ARCMIN2_SPHERE/npix*n_arcmin2

# the noise level from discrete observations with shape noise
nl = 4*np.pi*nbar/npix*sigma_e**2 * (l >= 2)

# mixing matrix for uniform distribution of points
mm = (nbar**2 - nbar/(npix-1))*np.eye(lmax+1, lmax+1) + (2*l+1)*nbar/(npix-1)/2
mm[:2, :] = mm[:, :2] = 0

# plot the realised and expected cls
plt.plot(l, cls[1] - nl, '-k', lw=2, label='simulation')
plt.plot(l, mm@(fl*theory_cls['W1xW1']), '-r', lw=2, label='expectation')
plt.xscale('symlog', linthresh=10, linscale=0.5, subs=[2, 3, 4, 5, 6, 7, 8, 9])
plt.yscale('symlog', linthresh=1e-9, linscale=0.5, subs=[2, 3, 4, 5, 6, 7, 8, 9])
plt.xlabel('angular mode number $l$')
plt.ylabel('angular power spectrum $C_l^{EE}$')
plt.legend()
plt.tight_layout()
plt.show()
