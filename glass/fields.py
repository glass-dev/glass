# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''
Random fields (:mod:`glass.fields`)
===================================

.. currentmodule:: glass.fields

The :mod:`glass.fields` module provides functionality for simulating random
fields on the sphere.  This is done in the form of HEALPix maps.

.. autosummary::
   :template: generator.rst
   :toctree: generated/
   :nosignatures:

   gaussian_gls
   lognormal_gls
   generate_gaussian
   generate_lognormal

'''

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
            if cl is None:
                cov[:, i] = 0
            else:
                n = len(cl)
                cov[:n, i] = cl
                cov[n:, i] = 0
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


def gaussian_gls(cls, *, lmax=None, ncorr=None, nside=None):
    '''Compute Gaussian Cls for a Gaussian random field.

    Depending on the given arguments, this truncates the angular power spectra
    to ``lmax``, removes all but ``ncorr`` correlations between fields, and
    applies the HEALPix pixel window function of the given ``nside``.  If no
    arguments are given, no action is performed.

    '''

    if ncorr is not None:
        n = int((2*len(cls))**0.5)
        if n*(n+1)//2 != len(cls):
            raise ValueError('length of cls array is not a triangle number')
        cls = [cls[i*(i+1)//2+j] if j <= ncorr else None for i in range(n) for j in range(i+1)]

    if nside is not None:
        pw = hp.pixwin(nside, lmax=lmax)

    gls = []
    for cl in cls:
        if cl is not None:
            if lmax is not None:
                cl = cl[:lmax+1]
            if nside is not None:
                n = min(len(cl), len(pw))
                cl = cl[:n] * pw[:n]**2
        gls.append(cl)
    return gls


def lognormal_gls(cls, shift=1., *, lmax=None, ncorr=None, nside=None):
    '''Compute Gaussian Cls for a lognormal random field.'''
    gls = gaussian_gls(cls, lmax=lmax, ncorr=ncorr, nside=nside)
    return transform_cls(gls, 'lognormal', (shift,))


def generate_gaussian(gls, nside, *, ncorr=None, rng=None):
    '''Iteratively sample Gaussian random fields from Cls.

    A generator that iteratively samples HEALPix maps of Gaussian random fields
    with the given angular power spectra ``gls`` and resolution parameter
    ``nside``.

    The optional argument ``ncorr`` can be used to artificially limit now many
    realised fields are correlated.  This saves memory, as only `ncorr` previous
    fields need to be kept.

    The ``gls`` array must contain the auto-correlation of each new field
    followed by the cross-correlations with all previous fields in reverse
    order::

        gls = [gl_00,
               gl_11, gl_10,
               gl_22, gl_21, gl_20,
               ...]

    Missing entries can be set to ``None``.

    '''

    # get the default RNG if not given
    if rng is None:
        rng = np.random.default_rng()

    # number of gls and number of fields
    ngls = len(gls)
    ngrf = int((2*ngls)**0.5)

    # number of correlated fields if not specified
    if ncorr is None:
        ncorr = ngrf - 1

    # number of modes
    n = max((len(gl) for gl in gls if gl is not None), default=0)
    if n == 0:
        raise ValueError('all gls are empty')

    # generates the covariance matrix for the iterative sampler
    cov = cls2cov(gls, n, ngrf, ncorr)

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


def generate_lognormal(gls, nside, shift=1., *, ncorr=None, rng=None):
    '''Iterative sample lognormal random fields from Gaussian Cls.'''
    for i, m in enumerate(generate_gaussian(gls, nside, ncorr=ncorr, rng=rng)):
        # compute the variance of the auto-correlation
        gl = gls[i*(i+1)//2]
        ell = np.arange(len(gl))
        var = np.sum((2*ell + 1)*gl)/(4*np.pi)

        # fix mean of the Gaussian random field for lognormal transformation
        m -= var/2

        # exponentiate values in place and subtract 1 in one operation
        np.expm1(m, out=m)

        # lognormal shift, unless unity
        if shift != 1:
            m *= shift

        # yield the lognormal map
        yield m
