'''
Galaxy distribution
===================

This example simulates a matter-only light cone up to redshift 1 and samples
galaxies from a uniform distribution in volume.  The results are shown in a
pseudo-3D plot.  This helps to make sure the galaxies sampling across shells
works as intended.

'''

# %%
# Setup
# -----
# Set up a galaxy positions-only GLASS simulation.  It needs very little:
# a way to obtain matter angular power spectra (here: CAMB) and a redshift
# distribution of galaxies to sample from (here: uniform in volume).

import numpy as np
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

# basic parameters of the simulation
nside = 128
lmax = nside

# galaxy density
n_arcmin2 = 0.01

# create a cosmology object
cosmo = LCDM(h=h, Om=(Oc + Ob))

# uniform (in volume) source distribution with given angular density
z = np.linspace(0, 1, 101)
dndz = n_arcmin2*cosmo.dvc(z)/cosmo.vc(z[-1])

# set up CAMB parameters for matter angular power spectrum
pars = camb.set_params(H0=100*h, omch2=Oc*h**2, ombh2=Ob*h**2)

# generators for a galaxies-only simulation
generators = [
    glass.sim.zspace(z[0], z[-1], dz=0.1),
    glass.matter.mat_wht_density(cosmo),
    glass.camb.camb_matter_cl(pars, lmax),
    glass.matter.lognormal_matter(nside),
    glass.galaxies.gal_dist_fullsky(z, dndz),
]


# %%
# Simulation
# ----------
# The goal of this example is to make a 3D cube of the sampled galaxy numbers.
# An comoving distance cube is initialised with zero counts, and the simulation
# is run.  For every shell in the light cone, the galaxies are counted in the
# cube.

# make a cube for galaxy number in comoving distance
xbin = cosmo.xc(z)
xbin = np.concatenate([-xbin[:0:-1], xbin])
cube = np.zeros((xbin.size-1,)*3)

# simulate and add galaxies in each matter shell to cube
for shell in glass.sim.generate(generators):
    rgal = cosmo.xc(shell['gal_z'])
    lon, lat = np.deg2rad(shell['gal_lon']), np.deg2rad(shell['gal_lat'])
    x1 = rgal*np.cos(lon)*np.cos(lat)
    x2 = rgal*np.sin(lon)*np.cos(lat)
    x3 = rgal*np.sin(lat)
    (i, j, k), c = np.unique(np.searchsorted(xbin[1:], [x1, x2, x3]), axis=1, return_counts=True)
    cube[i, j, k] += c


# %%
# Visualisation
# -------------
# Lastly, make a pseudo-3D plot by stacking a number of density slices on top of
# each other.

# positions of grid cells of the cube
x = (xbin[:-1] + xbin[1:])/2
x1, x2, x3 = np.meshgrid(x, x, x)

# plot the galaxy distribution in pseudo-3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d', proj_type='ortho')
vmin, vmax = 0, 0.8*np.max(cube)
for i in range(10, len(xbin)-1, 10):
    v = np.clip((cube[..., i] - vmin)/(vmax - vmin), 0, 1)
    c = plt.cm.inferno(v)
    c[..., -1] = 0.5*v
    ax.plot_surface(x1[..., i], x2[..., i], x3[..., i], facecolors=c, rstride=1, cstride=1, shade=False)
fig.tight_layout()
plt.show()
