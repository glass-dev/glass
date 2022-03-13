'''
Weak lensing
============

This example computes weak lensing maps (convergence and shear) for a redshift
distribution of sources.  The lensing is simulated by a line of sight
integration of the matter fields.

'''

# %%
# Setup
# -----
# To the simulate lensing fields in each shell of the light cone, it suffices to
# generate random (e.g. lognormal) matter fields, and perform a line of sight
# integration to obtain the convergence field.  The shear field is readily
# computed from there.
#
# From there, it is possible to obtain the effective integrated lensing maps of
# a distribution of sources.  Given such a distribution, which is set up here at
# the top, the :func:`glass.lensing.lensing_dist` generator will iteratively
# collect and integrate the contributions to the lensing from each shell.

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

# these are the GLASS imports: cosmology, glass modules, and the CAMB module
from cosmology import LCDM
import glass.sim
import glass.camb
import glass.matter
import glass.lensing

# also needs camb itself to get the parameter object, and the expectation
import camb


# cosmology for the simulation
cosmo = LCDM(h=0.7, Om=0.3)

# basic parameters of the simulation
nside = 512
lmax = nside

# localised redshift distribution
z = np.linspace(0, 1, 101)
nz = np.exp(-(z - 0.5)**2/(0.1)**2)

# set up CAMB parameters for matter angular power spectrum
pars = camb.set_params(H0=100*cosmo.h, omch2=cosmo.Om*cosmo.h**2,
                       NonLinear=camb.model.NonLinear_both)

# generators for a lensing-only simulation
generators = [
    glass.sim.zspace(0, 1.01, dz=0.1),
    glass.camb.camb_matter_cl(pars, lmax),
    glass.matter.lognormal_matter(nside),
    glass.lensing.convergence(cosmo),
    glass.lensing.shear(),
    glass.lensing.lensing_dist(z, nz, cosmo),
]


# %%
# Simulation
# ----------
# The simulation is then straightforward:  Only the integrated lensing maps are
# stored here.  While the simulation returns the result after every redshift
# interval in the light cone, only the last result will be show below, so the
# previous values are not kept.

# simulate and store the integrated lensing maps
for shell in glass.sim.generate(generators):
    kappa = shell['kappa_bar']
    gamma1 = shell['gamma1_bar']
    gamma2 = shell['gamma2_bar']


# %%
# Analysis
# --------
# To make sure the simulation works, compute the angular power spectrum ``cls``
# of the simulated convergence field, and compare with the expectation (from
# CAMB) for the given redshift distribution of sources.

# get the angular power spectra of the lensing maps
cls = hp.anafast([kappa, gamma1, gamma2], pol=True, lmax=lmax)

# get the expected cls from CAMB
pars.Want_CMB = False
pars.min_l = 1
pars.SourceWindows = [camb.sources.SplinedSourceWindow(z=z, W=nz, source_type='lensing')]
theory_cls = camb.get_results(pars).get_source_cls_dict(lmax=lmax, raw_cl=True)

# plot the realised and expected cls
l = np.arange(lmax+1)
plt.plot(l, (2*l+1)*cls[0], '-k', lw=2, label='simulation')
plt.plot(l, (2*l+1)*theory_cls['W1xW1'], '-r', lw=2, label='expectation')
plt.xscale('symlog', linthresh=10, linscale=0.5, subs=[2, 3, 4, 5, 6, 7, 8, 9])
plt.yscale('symlog', linthresh=1e-7, linscale=0.5, subs=[2, 3, 4, 5, 6, 7, 8, 9])
plt.xlabel(r'angular mode number $l$')
plt.ylabel(r'angular power spectrum $(2l+1) \, C_l^{\kappa\kappa}$')
plt.legend()
plt.show()
