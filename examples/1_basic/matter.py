'''
Matter distribution
===================

This example simulates only the matter field in nested shells up to redshift 1.

'''

# %%
# Setup
# -----
# Set up a matter-only GLASS simulation, which requires a way to obtain matter
# angular power spectra (here: CAMB) and the sampling itself (here: lognormal).

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
nside = 1024
lmax = nside
zend = 1.

# set up CAMB parameters for matter angular power spectrum
pars = camb.set_params(H0=100*h, omch2=Oc*h**2, ombh2=Ob*h**2)

# generators for a matter-only simulation
generators = [
    glass.sim.xspace(cosmo, 0, zend, dx=200.),
    glass.matter.mat_wht_density(cosmo),
    glass.camb.camb_matter_cl(pars, lmax),
    glass.matter.lognormal_matter(nside),
]


# %%
# Simulation
# ----------
# Run the simulation.  For each shell, plot an orthographic annulus of the
# matter distribution.

# make a 2d grid in comoving distance
# precompute the 2d radius r for the orthographic projection
n = 2000
rend = 1.05*cosmo.xc(zend)
x, y = np.mgrid[-rend:rend:1j*n, -rend:rend:1j*n]
r = np.hypot(x, y)
grid = np.full(r.shape, np.nan)

# set up the plot
ax = plt.subplot(111)
ax.axis('off')

# simulate and project an annulus of each matter shell onto the grid
for shell in glass.sim.generate(generators):
    rmin, rmax = cosmo.xc(shell['zmin', 'zmax'])
    delt = shell['delta']
    g = (rmin <= r) & (r < rmax)
    z = np.sqrt(1 - (r[g]/rmax)**2)
    theta, phi = hp.vec2ang(np.transpose([x[g]/rmax, y[g]/rmax, z]))
    grid[g] = hp.get_interp_val(delt, theta, phi)
    ax.add_patch(plt.Circle((0, 0), rmax, fc='none', ec='k', lw=0.5, alpha=0.5, zorder=1))

# show the grid of shells
ax.imshow(grid, extent=[-rend, rend, -rend, rend], zorder=0,
          cmap='bwr', vmin=-2, vmax=2)

# show the resulting plot
plt.show()
