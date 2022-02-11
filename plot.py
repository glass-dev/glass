import numpy as np
import healpy as hp
import fitsio

import matplotlib.pyplot as plt

from sortcl import enumerate_cls, cl_indices


def cls_from_pyccl(fields, lmax, zbins):
    '''compute theory cls with pyccl'''

    import pyccl

    c = pyccl.Cosmology(h=0.7, Omega_c=0.3, Omega_b=0.05, sigma8=0.8, n_s=0.96)

    zz = np.arange(0., zbins[-1]+0.001, 0.01)
    bz = np.ones_like(zz)
    ez = 1/c.h_over_h0(1/(1+zz))

    # physical types, either from dict or same as fields
    if isinstance(fields, dict):
        physs = [phys or field for field, phys in fields.items()]
    else:
        physs = fields

    # add tracers for every field based on its physical type
    names, tracers = [], []
    for field, phys in zip(fields, physs):
        for i, (za, zb) in enumerate(zip(zbins, zbins[1:])):
            nz = ((zz >= za) & (zz < zb))*ez

            if phys == 'matter':
                tracer = pyccl.NumberCountsTracer(c, False, (zz, nz), (zz, bz), None)
            elif phys == 'convergence':
                tracer = pyccl.WeakLensingTracer(c, (zz, nz), True, None, False)
            else:
                raise ValueError(f'cls_from_pyccl: unknown type "{phys}" for field "{field}"')

            names.append(f'{field}[{i}]')
            tracers.append(tracer)

    n = len(tracers)
    l = np.arange(lmax+1)

    cls = {}
    for i, j in zip(*cl_indices(n)):
        cls[names[i], names[j]] = pyccl.angular_cl(c, tracers[i], tracers[j], l)

    return cls


fits = fitsio.FITS('map.fits')

nbins = 10
lmax = 1000

zbins = []
for i in range(nbins):
    h = fits[f'MAP{i+1}'].read_header()
    if len(zbins) == 0:
        zbins.append(h['ZMIN'])
    else:
        assert zbins[-1] == h['ZMIN']
    zbins.append(h['ZMAX'])

delta = [fits[f'MAP{i+1}'].read(columns=['delta'])['delta'] for i in range(nbins)]
kappa = [fits[f'MAP{i+1}'].read(columns=['kappa'])['kappa'] for i in range(nbins)]

delta_cls = hp.anafast(delta, lmax=lmax, pol=False, use_pixel_weights=True)
kappa_cls = hp.anafast(kappa, lmax=lmax, pol=False, use_pixel_weights=True)

theory_cls = cls_from_pyccl({'delta': 'matter', 'kappa': 'convergence'}, lmax, zbins)

l = np.arange(lmax+1)

################################################################################

fig, ax = plt.subplots(nbins+1, nbins+1)

for i in range(nbins+1):
    ax[i, i].axis('off')
    ax[i, i].set_facecolor('grey')
    ax[i, i].add_artist(ax[i, i].patch)
    ax[i, i].patch.set_zorder(-1)

for i, j, cl in enumerate_cls(delta_cls):
    ax[i, j+1].plot(l, (2*l+1)*cl)
    ax[i, j+1].plot(l, (2*l+1)*theory_cls[f'delta[{i}]', f'delta[{j}]'])
    ax[i, j+1].set_xscale('symlog', linthresh=10, linscale=0.5)
    ax[i, j+1].set_yscale('symlog', linthresh=1e-4, linscale=0.5)
    ax[i, j+1].set_xlim(0, lmax)
    ax[i, j+1].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

for i, j, cl in enumerate_cls(kappa_cls):
    ax[j+1, i].plot(l, (2*l+1)*cl)
    ax[j+1, i].plot(l, (2*l+1)*theory_cls[f'kappa[{i}]', f'kappa[{j}]'])
    ax[j+1, i].set_xscale('symlog', linthresh=10, linscale=0.5)
    ax[j+1, i].set_yscale('symlog', linthresh=1e-7, linscale=0.5)
    ax[j+1, i].set_xlim(0, lmax)
    ax[j+1, i].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

fig.tight_layout(pad=0)

plt.show()
