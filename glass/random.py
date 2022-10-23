# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''simulation of spherical random fields'''

from itertools import repeat
import logging
import numpy as np
import healpy as hp
from gaussiancl import gaussiancl

log = logging.getLogger(__name__)


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


def transform_cls(cls, tfm, pars):
    '''transform Cls to Gaussian Cls for simulation'''

    # transform input cls to cls for the Gaussian random fields
    gls = []
    for cl in cls:
        # only work on available cls
        if cl is not None:
            log.info('computing Gaussian cl of size %d', len(cl))

            if cl[0] == 0:
                log.warning('warning: ignoring zero monopole')
                monopole = 0.
            else:
                monopole = False

            gl, info, err, niter = gaussiancl(cl, tfm, pars, monopole=monopole)

            if info == 0:
                log.warning('WARNING: solution did not converge, inexact transform')
            log.info('relative error after %d iterations: %g', niter, err)
        else:
            gl = None

        # store the Gaussian cl, or None
        gls.append(gl)

    # returns the list of transformed cls in input order
    return gls


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

        # transform to Gaussian cls
        cls = transform_cls(cls, 'normal', ())

        log.info('generating Gaussian random field')

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
        gls = transform_cls(cls, 'lognormal', (shift,))

        log.info('generating Gaussian random field')

        # get Gaussian random field for gls
        m = grf.send(gls)

        log.info('transforming to lognormal distribution')

        # fix mean of the Gaussian random field for lognormal transformation
        m -= np.dot(np.arange(1, 2*len(gls[0]), 2), gls[0])/(4*np.pi)/2

        # exponentiate values in place and subtract 1 in one operation
        np.expm1(m, out=m)

        # lognormal shift
        if shift != 1:
            m *= shift
