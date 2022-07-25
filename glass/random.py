# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''simulation of spherical random fields'''

import logging
import numpy as np
import healpy as hp
from gaussiancl import lognormal_cl


log = logging.getLogger(__name__)


def _exp_decay_cl(lmax, cl):
    '''exponential decay of power spectrum up to lmax'''
    l0 = len(cl)
    l = np.arange(l0, lmax+1)
    t = np.log(cl[-2]/cl[-1])/np.log((l0-2)/(l0-1))
    return np.concatenate([cl, cl[-1]*np.exp(t*(l/(l0-1) - 1))])


def synalm(cl, rng=None):
    '''sample spherical harmonic modes from an angular power spectrum

    This is a simplified version of the HEALPix synalm routine with support for
    random number generator instances.

    '''

    # get the default RNG if not given
    if rng is None:
        rng = np.random.default_rng()

    # length of the cl array
    n = len(cl)

    # sanity check
    if np.any(cl < 0):
        raise ValueError('negative values in cl')

    # standard normal random variates for alm
    # sample real and imaginary parts, then view as complex number
    alm = rng.standard_normal(n*(n+1), np.float64).view(np.complex128)

    # scale standard normal variates by cls
    # modes with m = 0 are first in array and real-valued
    f = np.sqrt(cl)/np.sqrt(2)
    for m in range(n):
        alm[m*n-m*(m-1)//2:(m+1)*n-m*(m+1)//2] *= f[m:]
    alm[:n].real += alm[:n].imag
    alm[:n].imag = 0

    # done with sampling alm
    return alm


def transform_cls(cls, tfm, nside=None):
    '''transform Cls to Gaussian Cls for simulation'''

    # maximum l in input cls
    lmax = np.max([len(cl)-1 for cl in cls if cl is not None])

    # map limit if nside is provided
    llim = int(12**0.5*nside - 1) if nside is not None else None

    # get pixel window function if nside is given, or set to unity
    if nside is not None:
        pw = hp.pixwin(nside, pol=False, lmax=llim)
    else:
        pw = np.ones(lmax+1)

    # the actual lmax is constrained by what the pixwin function can provide
    lmax = len(pw) - 1

    # transform input cls to cls for the Gaussian random fields
    gaussian_cls = []
    for cl in cls:
        # only work on available cls
        if cl is not None:
            # extrapolate the cl if possible
            if llim and llim >= len(cl):
                cl = _exp_decay_cl(llim, cl)

            # simulating integrated maps by multiplying cls and pw
            # shorter array determines length
            cl_len = min(len(cl), lmax+1)
            cl = cl[:cl_len]*pw[:cl_len]

            # transform the cl
            cl = tfm(cl)

        # store the Gaussian cl, or None
        gaussian_cls.append(cl)

    # returns the list of transformed cls in input order
    return gaussian_cls


def generate_gaussian(nside, rng=None):
    '''sample Gaussian random fields from Cls'''

    # initial values
    m = cl = alm = None

    # sample the fields from the conditional distribution
    while True:
        # yield random field and get cls for the next redshift slice
        try:
            cls = yield m
        except GeneratorExit:
            break

        # update the mean and variance of the conditional distribution
        # on first iteration, sample from unconditional distribution
        if cls[1] is not None:
            cl = cls[1]/np.where(cls[1] != 0, cl, 1)
            mu = hp.almxfl(alm, cl, inplace=True)
            cl = cls[0] - cls[1]*cl
        else:
            mu = 0
            cl = cls[0]

        # sample alm from the conditional distribution
        # does not include the mean, which is added below
        alm = synalm(cl, rng)

        # transform alm to maps
        # add the mean of the conditional distribution
        # can be performed in place on the temporary array
        m = hp.alm2map(alm + mu, nside, pixwin=False, pol=False, inplace=True)


def generate_normal(nside, rng=None):
    '''sample normal random fields from Cls'''

    # set up the underlying Gaussian random field generator
    grf = generate_gaussian(nside, rng)

    # prime generator
    m = grf.send(None)

    # sample each redshift slice
    while True:
        # yield lognormal field and get next cls
        try:
            cls = yield m
        except GeneratorExit:
            break

        # transform to Gaussian cls (applies pixel window function)
        cls = transform_cls(cls, lambda cl: cl, nside)

        # get Gaussian random field for cls
        m = grf.send(cls)


def generate_lognormal(nside, shift=1., rng=None):
    '''sample lognormal random fields from Cls'''

    # set up the underlying Gaussian random field generator
    grf = generate_gaussian(nside, rng)

    # prime generator
    m = grf.send(None)

    # sample each redshift slice
    while True:
        # yield lognormal field and get next cls
        try:
            cls = yield m
        except GeneratorExit:
            break

        # transform to Gaussian cls
        cls = transform_cls(cls, lambda cl: lognormal_cl(cl, alpha=shift), nside)

        # perform monopole surgery on cls
        if cls[0][0] < 0:
            log.info('monopole surgery required: epsilon = %.2e', -cls[0][0])
            if cls[0][0] < -1e-2:
                log.warn('warning: monopole surgery is causing significant changes')
            for cl in cls:
                if cl is not None:
                    cl[0] = 0

        # get Gaussian random field for cls
        m = grf.send(cls)

        # variance of Gaussian random field
        var = np.dot(np.arange(1, 2*len(cls[0]), 2), cls[0])/(4*np.pi)

        # fix mean of the Gaussian random field
        m += np.log(shift) - var/2

        # exponentiate values in place
        np.exp(m, out=m)

        # lognormal shift
        m -= shift
