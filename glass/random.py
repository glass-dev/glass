# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''simulation of spherical random fields'''

from itertools import repeat
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


def iternorm(k, cov, n=()):
    '''return the vector a and variance sigma^2 for iterative normal sampling'''

    m = np.zeros((*n, k, k))
    a = np.zeros((*n, k))
    s = np.zeros((*n,))
    q = (*n, k+1)
    j = 0 if k > 0 else None

    for i, x in enumerate(cov):
        x = np.asanyarray(x)
        if x.shape != q:
            try:
                x = np.broadcast_to(x, q)
            except ValueError:
                raise TypeError(f'covariance row {i}: shape {x.shape} cannot be broadcast to {q}') from None

        # only need to update matrix A if there are correlations
        if k > 0:
            # compute new entries of matrix A
            m[..., :, j] = 0
            m[..., [j], :] = np.matmul(a[..., np.newaxis, :], m)
            m[..., j, j] = np.where(s != 0, -1, 0)
            np.divide(m[..., j, :], -s[..., np.newaxis], where=(m[..., j, :] != 0), out=m[..., j, :])

            # compute new vector a
            c = x[..., 1:, np.newaxis]
            a = np.matmul(m[..., :j], c[..., k-j:, :])
            a += np.matmul(m[..., j:], c[..., :k-j, :])
            a = a.reshape(*n, k)

            # next rolling index
            j = (j - 1) % k

        # compute new standard deviation
        s = x[..., 0] - np.einsum('...i,...i', a, a)
        if np.any(s < 0):
            raise ValueError('covariance matrix is not positive definite')
        s = np.sqrt(s)

        # yield the next index, vector a, and standard deviation s
        yield j, a, s


def multalm(alm, bl, inplace=False):
    '''multiply alm by bl'''
    n = len(bl)
    if inplace:
        out = np.asanyarray(alm)
    else:
        out = np.copy(alm)
    for m in range(n):
        out[m*n-m*(m-1)//2:(m+1)*n-m*(m+1)//2] *= bl[m:]
    return out


def transform_cls(cls, tfm, nside=None):
    '''transform Cls to Gaussian Cls for simulation'''

    # maximum l in input cls
    lmax = max(len(cl)-1 for cl in cls if cl is not None)

    # bandlimit if nside is provided
    llim = 3*nside - 1 if nside is not None else None

    # get pixel window function if nside is given, or set to unity
    if nside is not None:
        log.info('using pixel window function for NSIDE=%d', nside)
        pw = hp.pixwin(nside, pol=False, lmax=llim)
        pw *= pw
    else:
        pw = np.ones(lmax+1)

    # transform input cls to cls for the Gaussian random fields
    gaussian_cls = []
    for cl in cls:
        # only work on available cls
        if cl is not None:
            # extrapolate the cl if necessary
            if llim and llim >= len(cl):
                log.info('exponentially decaying cl from %d to %d', len(cl)-1, llim)
                cl = _exp_decay_cl(llim, cl)

            # simulating integrated maps by multiplying cls and pw
            # shorter array determines length
            n = min(len(cl), len(pw))
            cl = cl[:n]*pw[:n]

            log.info('transforming cl with LMAX=%d', n-1)

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

    # initial value
    m = None

    # sample the fields from the conditional distribution
    while True:
        # yield random field and get cls for the next redshift slice
        try:
            cls = yield m
        except GeneratorExit:
            break

        # set up the iterative sampling on the first pass
        if m is None:
            k = len(cls) - 1
            n = len(cls[0])
            cov = np.zeros((n, k + 1))
            z = np.zeros(n*(n+1)//2, dtype=np.complex128)
            y = np.zeros((n*(n+1)//2, k), dtype=np.complex128)
            ni = iternorm(k, repeat(cov), (n,))

        # set the new covariance matrix row
        for i, cl in enumerate(cls):
            if i == 0 and np.any(cl < 0):
                raise ValueError('negative values in cl')
            cov[..., i] = cl if cl is not None else 0

        # covariance is per component
        cov /= 2

        # get the conditional distribution
        j, a, s = next(ni)

        # standard normal random variates for alm
        # sample real and imaginary parts, then view as complex number
        rng.standard_normal(n*(n+1), np.float64, z.view(np.float64))

        # scale by standard deviation of the conditional distribution
        # variance is distributed over real and imaginary part
        alm = multalm(z, s)

        # add the mean of the conditional distribution
        for i in range(k):
            alm += multalm(y[:, i], a[:, i])

        # store the standard normal in y array at the indicated index
        if j is not None:
            y[:, j] = z

        # modes with m = 0 are real-valued and come first in array
        alm[:n].real += alm[:n].imag
        alm[:n].imag = 0

        # transform alm to maps
        # can be performed in place on the temporary alm array
        m = hp.alm2map(alm, nside, pixwin=False, pol=False, inplace=True)


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
