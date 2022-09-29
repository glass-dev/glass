'''
Stage IV Galaxy Survey
======================

This example simulates a galaxy catalogue from a Stage IV Space Satellite Galaxy
Survey such as *Euclid* and *Roman* combining the
:ref:`sphx_glr_examples_1_basic_plot_density.py` and
:ref:`sphx_glr_examples_1_basic_plot_lensing.py` examples with generators for
the intrinsic galaxy ellipticity and the resulting shear with some auxiliary
functions.

The focus in this example is mock catalogue generation using auxiliary functions
built for simulating Stage IV galaxy surveys.
'''

# %%
# Setup
# -----
# The basic setup of galaxies and weak lensing fields is the same as in the
# previous examples.
#
# In addition to a generator for intrinsic galaxy ellipticities,
# following a normal distribution, we also show how to use auxiliary functions
# to generate tomographic redshift distributions and visibility masks.
#
# Finally, there is a generator that applies the reduced shear from the lensing
# maps to the intrinsic ellipticities, producing the galaxy shears.

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

# these are the GLASS imports: cosmology and the glass meta-module
from cosmology import LCDM
from glass import glass

# also needs camb itself to get the parameter object
import camb

# cosmology for the simulation
h = 0.7
Oc = 0.25
Ob = 0.05
cosmo = LCDM(h=h, Om=Oc+Ob)

# basic parameters of the simulation
nside = 512
lmax = nside

# comoving distance size in Mpc of each shell to integrate along the LoS:
dx = 200

# galaxy density (using 1/100 of the expected galaxy number density for Stage-IV)
n_arcmin2 = 0.3

# sigma_ellipticity as expected for a stage-IV survey
sigma_e = 0.27

# photometric redshift error
sigma_z0 = 0.03

# set up CAMB parameters for matter angular power spectrum
pars = camb.set_params(H0=100*h, omch2=Oc*h**2, ombh2=Ob*h**2)


# %%
# Simulation Setup
# ----------------
# Here we setup the overall source redshift distribution
# and separate it into equal density tomographic bins
# with the typical redshift errors of a photometric survey.

# setting up the random number generator:
rng = np.random.default_rng(seed=42)

# true redshift distribution following a Smail distribution
z = np.linspace(0, 3, 1000)
dndz = glass.observations.smail_nz(z, z_mode=0.9, alpha=2., beta=1.5)
dndz *= n_arcmin2
bz = 1.2

# compute bin edges with equal density
# then make tomographic bins, assuming photometric redshift errors
nbins = 10
zedges = glass.observations.equal_dens_zbins(z, dndz, nbins=nbins)
bin_nz = glass.observations.tomo_nz_gausserr(z, dndz, sigma_z0, zedges)

# %%
# Plotting the overall redshift distribution and the
# distribution for each of the equal density tomographic bins
plt.figure()
plt.title('redshift distributions')
sum_nz = np.zeros_like(bin_nz[0])
for nz in bin_nz:
    plt.fill_between(z, nz, alpha=0.5)
    sum_nz = sum_nz + nz
plt.fill_between(z, dndz, alpha=0.2, label='dn/dz')
plt.plot(z, sum_nz, ls='--', label="Sum of the bins")
plt.ylabel("dN/dz - gal/arcmin2")
plt.xlabel("z")
plt.legend()
plt.tight_layout()
plt.show()

# %%
# Make a visibility map typical of a space telescope survey, seeing both
# hemispheres, and low visibility in the galactic and ecliptic bands.
vis = glass.observations.vmap_galactic_ecliptic(nside)

# checking the mask:
hp.mollview(vis, title='Stage IV Space Survey-like Mask', unit='Visibility')
plt.show()

# %%
# generators for the clustering and lensing
generators = [
    glass.sim.xspace(cosmo, 0., 3., dx=dx),
    glass.matter.mat_wht_redshift(),
    glass.camb.camb_matter_cl(pars, lmax),
    glass.matter.lognormal_matter(nside, rng=rng),
    glass.lensing.convergence(cosmo),
    glass.lensing.shear(lmax),
    glass.observations.vis_constant(vis, nside=nside),
    glass.galaxies.gal_dist_fullsky(z, bin_nz, bz=bz, rng=rng),
    glass.galaxies.gal_ellip_gaussian(sigma_e, rng=rng),
    glass.galaxies.gal_shear_interp(cosmo),
]

# %%
# Simulation
# ----------
# Simulate the galaxies with shears.  In each iteration, get the quantities of interest
# to build our mock catalogue.

# we will store the catalogue as a dictionary:
catalogue = {'RA': np.array([]), 'DEC': np.array([]), 'TRUE_Z': np.array([]),
             'E1': np.array([]), 'E2': np.array([]), 'TOMO_ID': np.array([])}

# iterate and store the quantities of interest for our mock catalogue:
for shell in glass.sim.generate(generators):
    # let's assume here that lon lat here are RA and DEC:
    catalogue['RA'] = np.append(catalogue['RA'], shell['gal_lon'])
    catalogue['DEC'] = np.append(catalogue['DEC'], shell['gal_lat'])
    catalogue['TRUE_Z'] = np.append(catalogue['TRUE_Z'], shell['gal_z'])
    catalogue['E1'] = np.append(catalogue['E1'], shell['gal_ell'].real)
    catalogue['E2'] = np.append(catalogue['E2'], shell['gal_ell'].imag)
    catalogue['TOMO_ID'] = np.append(catalogue['TOMO_ID'], shell['gal_pop'])

print(f"Total Number of galaxies sampled: {len(catalogue['TRUE_Z']):,}")

# %%
# Catalogue checks
# ----------------
# Here we can perform some simple checks at the catalogue level to
# see how our simulation performed.

# redshift distribution of tomographic bins & input distributions
plt.figure()
plt.title('redshifts in catalogue')
plt.ylabel("dN/dz - normalised")
plt.xlabel("z")
for i in range(0, 10):
    plt.hist(catalogue['TRUE_Z'][catalogue['TOMO_ID'] == i], histtype='stepfilled', edgecolor='none', alpha=0.5, bins=50, density=1, label=f'cat. bin {i}')
for i in range(0, 10):
    plt.plot(z, (bin_nz[i]/n_arcmin2)*nbins, alpha=0.5, label=f'inp. bin {i}')
plt.plot(z, dndz/n_arcmin2*nbins, ls='--', c='k')
plt.legend(ncol=2)
plt.show()
