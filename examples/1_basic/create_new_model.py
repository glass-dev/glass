'''
Creating a new model
====================

This example shows how to implement and use a new model.

'''

# %%
# Setup
# -----
# The setup here is the same as in the :ref:`sphx_glr_examples_1_basic_plot_density.py`
# example, except that we use fewer galaxies and only a single matter shell.

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

# these are the GLASS imports: cosmology and the glass meta-module
from glass import glass

# also needs camb itself to get the parameter object
import camb

# this decorator marks generators in GLASS
from glass.core import generator


# cosmology for the simulation
h = 0.7
Oc = 0.25
Ob = 0.05

# basic parameters of the simulation
nside = 128
lmax = nside

# galaxy density
n_arcmin2 = 1e-5

# flat source distribution with given angular density
z = np.linspace(0, 0.1, 11)
dndz = np.ones_like(z)
dndz *= n_arcmin2/np.trapz(dndz, z)

# set up CAMB parameters for matter angular power spectrum
pars = camb.set_params(H0=100*h, omch2=Oc*h**2, ombh2=Ob*h**2)

# generators for a galaxies-only simulation
generators = [
    glass.sim.zspace(z[0], z[-1]+0.01, dz=0.1),
    glass.matter.mat_wht_redshift(),
    glass.camb.camb_matter_cl(pars, lmax),
    glass.matter.lognormal_matter(nside),
    glass.galaxies.gal_dist_fullsky(z, dndz),
]

# %%
# Model
# -----
# We will now add a custom model that returns a flag to indicate whether a
# galaxy is found in a matter overdensity or underdensity.
#
# The new model is prototypical of "inputs to outputs" models in GLASS.  Like
# all GLASS models, it is implemented as a simple Python generator.  The model
# can take global parameters, here a ``thresh`` value for the threshold at
# which matter is considered overdense.  The generator then runs in a loop: At
# each iteration of the simulation, it receives the matter density ``delta``
# and galaxy positions ``gal_lon, gal_lat``.  It then yields the newly computed
# overdensity flag ``gal_od_flag`` provided by the model.


# the decorator marks this as a GLASS generator
# its signature determines the inputs and outputs
@generator('delta, gal_lon, gal_lat -> gal_od_flag')
def gal_od_flag_model(thresh=0.):

    # initial yield
    od_flag = None

    # receive inputs and yield outputs, or break on exit
    while True:
        try:
            delta, lon, lat = yield od_flag
        except GeneratorExit:
            break

        # perform the computation for this iteration
        # get the HEALPix pixel index of the galaxies
        # set the flag according to whether the overdensity is above threshold
        nside = hp.get_nside(delta)
        ipix = hp.ang2pix(nside, lon, lat, lonlat=True)
        od = delta[ipix]
        od_flag = (od > thresh)

        # the computation then loops around and yields our latest results

    # it's possible to post-process after the simulation stopped
    print('done with our model')


# add our new model to the generators used in the simulation
generators.append(gal_od_flag_model(thresh=0.01))

# %%
# Simulation
# ----------
# Run the simulation.  We will keep track of galaxy positions and their
# overdensity flags returned by our model.

# keep lists of positions and the overdensity flags
lon, lat, od_flag = np.empty(0), np.empty(0), np.empty(0, dtype=bool)

# simulate and add galaxies in each iteration to lists
for it in glass.sim.generate(generators):
    lon = np.append(lon, it['gal_lon'])
    lat = np.append(lat, it['gal_lat'])
    od_flag = np.append(od_flag, it['gal_od_flag'])


# %%
# Visualisation
# -------------
# Show the positions of galaxies in underdense regions.

plt.subplot(111, projection='lambert')
plt.title('galaxies in underdensities')
plt.scatter(np.deg2rad(lon[~od_flag]), np.deg2rad(lat[~od_flag]), 8.0, 'r', alpha=0.5)
plt.grid(True)
plt.show()
