# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''simulation of spherical random fields'''

import logging
import numpy as np
import healpy as hp
from gaussiancl import lognormal_cl


log = logging.getLogger(__name__)


def transform_cls(cls, tfm, nside=None):
    '''transform Cls to Gaussian Cls for simulation'''

    # maximum l in input cls
    lmax = np.max([len(cl)-1 for cl in cls if cl is not None])

    # get pixel window function if nside is given, or set to unity
    if nside is not None:
        pw = hp.pixwin(nside, pol=False, lmax=lmax)
    else:
        pw = np.ones(lmax+1)

    # the actual lmax is constrained by what the pixwin function can provide
    lmax = len(pw) - 1

    # transform input cls to cls for the Gaussian random fields
    gaussian_cls = []
    for cl in cls:
        # only work on available cls
        if cl is not None:
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

    # get the default RNG if not given
    if rng is None:
        rng = np.random.default_rng()

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
            cl = cls[1]/cl
            mu = hp.almxfl(alm, cl, inplace=True)
            cl = cls[0] - cls[1]*cl
        else:
            mu = 0
            cl = cls[0]

        # sample alm from the conditional distribution
        # does not include the mean, which is added below
        alm = hp.synalm(cl)

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
