import numpy as np
import healpy as hp
import fitsio

import matplotlib.pyplot as plt

from sortcl import enumerate_cls

import camb


nbins = 10
lmax = 1024

l = np.arange(lmax+1)

################################################################################

fits = fitsio.FITS('map.fits')

zbins = []
for i in range(nbins):
    h = fits[f'MAP{i+1}'].read_header()
    zbins.append((h['ZMIN'], h['ZMAX']))

delta = [fits[f'MAP{i+1}'].read(columns=['delta'])['delta'] for i in range(nbins)]
kappa = [fits[f'MAP{i+1}'].read(columns=['kappa'])['kappa'] for i in range(nbins)]

delta_cls = hp.anafast(delta, lmax=lmax, pol=False, use_pixel_weights=True)
kappa_cls = hp.anafast(kappa, lmax=lmax, pol=False, use_pixel_weights=True)

################################################################################

pars = camb.set_params(H0=70, omch2=0.3*(0.7)**2)

pars.Want_CMB = False
pars.Want_Cls = True
pars.SourceTerms.counts_density = True
pars.SourceTerms.counts_evolve = True
pars.SourceTerms.counts_redshift = False
pars.SourceTerms.counts_lensing = False
pars.SourceTerms.counts_velocity = False
pars.SourceTerms.counts_radial = False
pars.SourceTerms.counts_timedelay = False
pars.SourceTerms.counts_ISW = False
pars.SourceTerms.counts_potential = False

results = camb.get_background(pars, no_thermo=True)

windows = []
for i, (zmin, zmax) in enumerate(zbins):
    z = np.linspace(zmin, zmax, 100)
    w = ((1 + z)*results.angular_diameter_distance(z))**2/results.h_of_z(z)
    w /= np.trapz(w, z)
    windows.append(camb.sources.SplinedSourceWindow(source_type='counts', z=z, W=w))
    windows.append(camb.sources.GaussianSourceWindow(source_type='lensing', redshift=zmax, sigma=0.01))

pars.SourceWindows = windows

results = camb.get_results(pars)
camb_cls = results.get_source_cls_dict(lmax=lmax, raw_cl=True)

################################################################################

fig, ax = plt.subplots(nbins+1, nbins+1)

for i in range(nbins+1):
    ax[i, i].axis('off')
    ax[i, i].set_facecolor('grey')
    ax[i, i].add_artist(ax[i, i].patch)
    ax[i, i].patch.set_zorder(-1)

for i, j, cl in enumerate_cls(delta_cls):
    ax[i, j+1].plot(l, (2*l+1)*cl)
    ax[i, j+1].plot(l, (2*l+1)*camb_cls[f'W{2*i+1}xW{2*j+1}'])
    ax[i, j+1].set_xscale('symlog', linthresh=10, linscale=0.5)
    ax[i, j+1].set_yscale('symlog', linthresh=1e-4, linscale=0.5)
    ax[i, j+1].set_xlim(0, lmax)
    ax[i, j+1].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

for i, j, cl in enumerate_cls(kappa_cls):
    ax[j+1, i].plot(l, (2*l+1)*cl)
    ax[j+1, i].plot(l, (2*l+1)*camb_cls[f'W{2*i+2}xW{2*j+2}'])
    ax[j+1, i].set_xscale('symlog', linthresh=10, linscale=0.5)
    ax[j+1, i].set_yscale('symlog', linthresh=1e-7, linscale=0.5)
    ax[j+1, i].set_xlim(0, lmax)
    ax[j+1, i].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

fig.tight_layout(pad=0)

plt.show()
