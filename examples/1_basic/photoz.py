'''
Photometric redshifts
=====================

This example simulates galaxies with a simple photometric redshift model.

'''

# %%
# Setup
# -----
# Set up a galaxy positions-only GLASS simulation.  It needs very little:
# a way to obtain matter angular power spectra (here: CAMB) and a redshift
# distribution of galaxies to sample from (here: uniform in volume).  Then add
# a model for photometric redshifts with Gaussian errors.

import numpy as np
import matplotlib.pyplot as plt

# these are the GLASS imports: cosmology and the glass meta-module
from cosmology import LCDM
from glass import glass

# also needs camb itself to get the parameter object
import camb


# cosmology for the simulation
cosmo = LCDM(h=0.7, Om=0.3)

# basic parameters of the simulation
nside = 128
lmax = nside

# galaxy density
n_arcmin2 = 1e-4

# photometric redshift error at redshift 0
phz_sigma_0 = 0.05

# parametric galaxy redshift distribution
z = np.linspace(0, 3, 301)
dndz = n_arcmin2*glass.observations.smail_nz(z, 1.0, 2.2, 1.5)

# set up CAMB parameters for matter angular power spectrum
pars = camb.set_params(H0=100*cosmo.h, omch2=cosmo.Om*cosmo.h**2)

# generators for a galaxies-only simulation
generators = [
    glass.sim.zspace(z[0], z[-1]+0.01, dz=0.25),
    glass.camb.camb_matter_cl(pars, lmax),
    glass.matter.lognormal_matter(nside),
    glass.galaxies.gal_dist_fullsky(z, dndz),
    glass.galaxies.gal_phz_gausserr(phz_sigma_0),
]


# %%
# Simulation
# ----------
# Keep all simulated true and photometric redshifts.

# arrays for true (ztrue) and photmetric (zphot) redshifts
ztrue = np.empty(0)
zphot = np.empty(0)

# simulate and add galaxies in each matter shell to cube
for shell in glass.sim.generate(generators):
    ztrue = np.append(ztrue, shell['gal_z'])
    zphot = np.append(zphot, shell['gal_z_phot'])


# %%
# Plots
# -----
# Make a couple of typical photometric redshift plots.
#
# First the :math:`z`-vs-:math:`z` plot across the entire sample.  The simple
# Gaussian error model only has the diagonal but no catastrophic outliers.

plt.figure(figsize=(5, 5))
plt.plot(ztrue, zphot, '+k', ms=3, alpha=0.1)
plt.xlabel(r'$z_{\rm true}$', size=12)
plt.ylabel(r'$z_{\rm phot}$', size=12)
plt.show()

# %%
# Now define a number of photometric redshift bins.  They are chosen by the
# :func:`~glass.observations.equal_dens_zbins` function to produce the same
# number of galaxies in each bin.

nbins = 5
zbins = glass.observations.equal_dens_zbins(z, dndz, nbins)

# %%
# After the photometric bins are defined, make histograms of the *true* redshift
# distribution :math:`n(z)` using the *photometric* redshifts for binning.
#
# Use the :func:`~glass.observations.tomo_nz_gausserr()` to get the expected
# tomographic redshift distributions using the same Gaussian error model.

tomo_nz = glass.observations.tomo_nz_gausserr(z, dndz, phz_sigma_0, zbins)
tomo_nz *= glass.util.ARCMIN2_SPHERE*(z[-1] - z[0])/40

for (z1, z2), nz in zip(zbins, tomo_nz):
    plt.hist(ztrue[(z1 <= zphot) & (zphot < z2)], bins=40, range=(z[0], z[-1]),
             histtype='stepfilled', alpha=0.5)
    plt.plot(z, nz, '-k', lw=1, alpha=0.5)
plt.xlabel('true redshift $z$')
plt.ylabel('number of galaxies')
plt.show()
