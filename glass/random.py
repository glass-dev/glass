# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''simulation of spherical random fields'''

from numbers import Number
import warnings
import numpy as np
import healpy as hp
from gaussiancl import gaussiancl


def iternorm(k, cov, size=None):
    '''return the vector a and variance sigma^2 for iterative normal sampling'''

    if size is None:
        n = ()
    elif isinstance(size, Number):
        n = (size,)
    else:
        n = size

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
            m[..., j:j+1, :] = np.matmul(a[..., np.newaxis, :], m)
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


def cls2cov(cls, nl, nf, nc):
    '''Return array of cls as a covariance matrix for iterative sampling.'''
    cov = np.zeros((nl, nc+1))
    end = 0
    for j in range(nf):
        begin, end = end, end + j + 1
        for i, cl in enumerate(cls[begin:end][:nc+1]):
            if i == 0 and np.any(cl < 0):
                raise ValueError('negative values in cl')
            cov[:, i] = cl if cl is not None else 0
        cov /= 2
        yield cov


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


def transform_cls(cls, tfm, pars=()):
    '''Transform Cls to Gaussian Cls.'''

    gls = []
    for cl in cls:
        if cl is not None:
            if cl[0] == 0:
                monopole = 0.
            else:
                monopole = None

            gl, info, err, niter = gaussiancl(cl, tfm, pars, monopole=monopole)

            if info == 0:
                warnings.warn('Gaussian cl did not converge, inexact transform')
        else:
            gl = None

        gls.append(gl)

    return gls


def generate_grf(cls, nside, ncorr=None, *, rng=None):
    '''Iteratively sample Gaussian random fields from Cls.

    A generator that iteratively samples HEALPix maps of Gaussian random fields
    with the given angular power spectra ``cls`` and resolution parameter
    ``nside``.

    The optional argument ``ncorr`` can be used to artificially limit now many
    realised fields are correlated.  This saves memory, as only `ncorr` previous
    fields need to be kept.

    The ``cls`` array must contain the auto-correlation of each new field
    followed by the cross-correlations with all previous fields in reverse
    order::

        cls = [cl_00,
               cl_11, cl_10,
               cl_22, cl_21, cl_20,
               ...]

    Missing entries can be set to ``None``.

    '''

    # get the default RNG if not given
    if rng is None:
        rng = np.random.default_rng()

    # number of cls and number of fields
    ncls = len(cls)
    ngrf = int((2*ncls)**0.5)

    # number of correlated fields if not specified
    if ncorr is None:
        ncorr = ngrf - 1

    # number of modes
    n = max((len(cl) for cl in cls if cl is not None), default=0)
    if n == 0:
        raise ValueError('all cls are empty')

    # generates the covariance matrix for the iterative sampler
    cov = cls2cov(cls, n, ngrf, ncorr)

    # working arrays for the iterative sampling
    z = np.zeros(n*(n+1)//2, dtype=np.complex128)
    y = np.zeros((n*(n+1)//2, ncorr), dtype=np.complex128)

    # generate the conditional normal distribution for iterative sampling
    conditional_dist = iternorm(ncorr, cov, size=n)

    # sample the fields from the conditional distribution
    for j, a, s in conditional_dist:
        # standard normal random variates for alm
        # sample real and imaginary parts, then view as complex number
        rng.standard_normal(n*(n+1), np.float64, z.view(np.float64))

        # scale by standard deviation of the conditional distribution
        # variance is distributed over real and imaginary part
        alm = multalm(z, s)

        # add the mean of the conditional distribution
        for i in range(ncorr):
            alm += multalm(y[:, i], a[:, i])

        # store the standard normal in y array at the indicated index
        if j is not None:
            y[:, j] = z

        # modes with m = 0 are real-valued and come first in array
        alm[:n].real += alm[:n].imag
        alm[:n].imag = 0

        # transform alm to maps
        # can be performed in place on the temporary alm array
        yield hp.alm2map(alm, nside, pixwin=False, pol=False, inplace=True)


def generate_normal(cls, nside, ncorr=None, *, rng=None):
    '''sample normal random fields from Cls'''

    # transform to Gaussian cls
    gls = transform_cls(cls, 'normal')

    # sample maps of Gaussian random fields, no processing needed
    for m in generate_grf(gls, nside, ncorr, rng=rng):
        yield m


def generate_lognormal(cls, nside, shift=1., ncorr=None, *, rng=None):
    '''sample lognormal random fields from Cls'''

    # transform to Gaussian cls
    gls = transform_cls(cls, 'lognormal', (shift,))

    # sample maps of Gaussian random fields and transform to lognormal
    for m in generate_grf(gls, nside, ncorr, rng=rng):
        # fix mean of the Gaussian random field for lognormal transformation
        m -= np.var(m)/2

        # exponentiate values in place and subtract 1 in one operation
        np.expm1(m, out=m)

        # lognormal shift, unless unity
        if shift != 1:
            m *= shift

        # yield the lognormal map
        yield m
