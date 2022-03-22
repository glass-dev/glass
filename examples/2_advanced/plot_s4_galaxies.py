'''
Stage IV Space Satellite Galaxy Survey
============

This example simulates a galaxy catalogue from a Stage IV Space Satellite Galaxy Survey such as
*Euclid* and *Roman* combining the :ref:`sphx_glr_examples_1_basic_plot_density.py` and
:ref:`sphx_glr_examples_1_basic_plot_lensing.py` examples with generators for
the intrinsic galaxy ellipticity and the resulting shear with some auxiliary functions.

The focus in this example is mock catalogue generation using auxiliary functions
built for simulating Stage-IV galaxies.
'''

# %%
# Setup
# -----
# The basic setup of galaxies and weak lensing fields is the same as in the
# previous examples.
#
# In addition to a generator for intrinsic galaxy ellipticities,
# following a normal distribution, we also show how to use auxiliary functions
# to generate photometric redshift distributions and visibility masks.
#
# Finally, there is a generator that applies the reduced shear from the lensing
# maps to the intrinsic ellipticities, producing the galaxy shears.

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

# these are the GLASS imports: cosmology and glass ifself
from cosmology import LCDM
from glass import glass

# also needs camb itself to get the parameter object
import camb

# cosmology for the simulation
cosmo = LCDM(h=0.7, Om=0.3)

# basic parameters of the simulation
nside = 512
lmax = nside

# size of the dz of each shell to integrate along the LoS:
dz = 0.05

# galaxy density (using 1/100 of the expected galaxy number density for Stage-IV)
n_arcmin2 = 0.3

# sigma_ellipticity as expected for a stage-IV survey
sigma_e = 0.27

# photometric redshift error
sigma_z0 = 0.03

# set up CAMB parameters for matter angular power spectrum
pars = camb.set_params(H0=100*cosmo.h, omch2=cosmo.Om*cosmo.h**2)

# %%
# Simulation Setup
# ----------------
# Here we setup the overall photometric redshift distribution
# and separate it into equal density tomographic bins
# with photometric redshift errors.

# setting up the random number generator:
rng = np.random.default_rng(seed=42)

# photometric redshift distribution following a Smail distribution
z = np.linspace(0, 3.0, 1000)
dndz = glass.observations.smail_nz(z, z_mode=0.9, alpha=2., beta=1.5)
dndz *= n_arcmin2
bz = 1.2

# equal density bins:
nbins = 10
zedges = glass.observations.equal_dens_zbins(z, dndz, nbins=nbins)
bin_nz = glass.observations.tomo_nz_gausserr(z, dndz, sigma_z0, zedges)

# %%
# Plotting the overall redshift distribution and the
# distribution for each of the equal density tomographic bins
plt.ion()
plt.figure(figsize=(10, 5))
plt.title("Stage IV Space Telescope - Photometric Distribution: equal density bins")
SumNz = np.zeros_like(bin_nz[0])
for nz in bin_nz:
    plt.fill_between(z, nz, alpha=0.5)
    SumNz = SumNz + nz
plt.fill_between(z, dndz, alpha=0.2, label='dn/dz')
plt.plot(z, SumNz, ls='--', label="Sum of the bins")
plt.ylabel("dN/dz - gal/arcmin2")
plt.xlabel("z")
plt.legend()
plt.tight_layout()
plt.pause(1e-3)

# %%
# Make a visibility map with low NSIDE
# also compute its fsky for the extected galaxy count
stageIV_mask = glass.observations.vmap_galactic_ecliptic(nside)

# checking the mask:
hp.mollview(stageIV_mask, title='Stage IV Space Survey-like Mask', unit='Visibility')
plt.pause(1e-3)

# %%
# generators for the clustering and lensing
generators = [
    glass.sim.zspace(0., 3.0001, dz=dz),
    glass.camb.camb_matter_cl(pars, lmax),
    glass.matter.lognormal_matter(nside, rng=rng),
    glass.lensing.convergence(cosmo),
    glass.lensing.shear(lmax),
    glass.observations.vis_constant(stageIV_mask, nside=nside),
    glass.galaxies.gal_dist_fullsky(z, bin_nz, bz=bz, rng=rng),
    glass.galaxies.gal_ellip_gaussian(sigma_e, rng=rng),
    glass.galaxies.gal_shear_interp(cosmo),
]

# %%
# Simulation
# ----------
# Simulate the galaxies with shears.  In each iteration, get the quantities of interest
# to build our mock catalogue.

# keep count of total number of galaxies
num = 0

# we will store the catalogue as a dictionary:
catalogue = {'RA': np.array([]), 'DEC': np.array([]), 'TRUE_Z': np.array([]),
             'E1': np.array([]), 'E2': np.array([]), 'TOMO_ID': np.array([])}

# iterate and store the quantities of interest for our mock catalogue:
for shell in glass.sim.generate(generators):
    print(f"Generating shell #: {shell['#']}")
    num += shell['ngal']
    # let's assume here that lon lat here are RA and DEC:
    catalogue['RA'] = np.append(catalogue['RA'], shell['gal_lon'])
    catalogue['DEC'] = np.append(catalogue['DEC'], shell['gal_lat'])
    catalogue['TRUE_Z'] = np.append(catalogue['TRUE_Z'], shell['gal_z'])
    catalogue['E1'] = np.append(catalogue['E1'], shell['gal_ell'].real)
    catalogue['E2'] = np.append(catalogue['E2'], shell['gal_ell'].imag)
    catalogue['TOMO_ID'] = np.append(catalogue['TOMO_ID'], shell['gal_pop'])

print(f"Total Number of galaxies sampled: {num}")

# %%
# Catalogue checks
# --------
# Here we can perform some simple checks at the catlaogue legal to
# see how our simulation performed.

# redshift distribution of tomographic bins & input distributions
plt.figure(figsize=(10, 5))
plt.title("Stage IV Space Telescope - Catalogue's Photometric Distribution")
plt.ylabel("dN/dz - normalised")
plt.xlabel("z")
[plt.hist(catalogue['TRUE_Z'][catalogue['TOMO_ID'] == i], edgecolor='black', alpha=0.4,
          bins=50, density=1, label=f'Catalogue Bin-{i}') for i in range(0, 10)]
[plt.fill_between(z, (bin_nz[i]/n_arcmin2)*nbins, alpha=0.6, hatch='//', label=f'Input bin-{i}') for i in range(0, 10)]
plt.plot(z, dndz/n_arcmin2*nbins, ls='--', c='k')
plt.legend(ncol=2)
plt.ioff()
plt.show()
