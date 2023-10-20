# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''
Random fields (:mod:`glass.fields`)
===================================

.. currentmodule:: glass.fields

The :mod:`glass.fields` module provides functionality for simulating random
fields on the sphere.  This is done in the form of HEALPix maps.

Angular power spectra
---------------------

.. autofunction:: discretized_cls
.. autofunction:: biased_cls
.. autofunction:: getcl

Generating random fields
------------------------

.. autofunction:: generate_alms

Gaussian random fields
----------------------

.. autofunction:: alm_to_gaussian
.. autofunction:: generate_gaussian

Lognormal fields
----------------

.. autofunction:: lognormal_gls
.. autofunction:: alm_to_lognormal
.. autofunction:: generate_lognormal


'''

import warnings
import numpy as np
import healpy as hp
from gaussiancl import gaussiancl

# typing
from typing import (Any, Union, Tuple, Generator, Optional, Sequence, Callable,
                    Iterable)
from numpy.typing import ArrayLike, NDArray

from .core import update_metadata

# types
Array = NDArray
Size = Union[None, int, Tuple[int, ...]]
Iternorm = Tuple[Optional[int], Array, Array]
ClTransform = Union[str, Callable[[Array], Array]]
Cls = Sequence[Union[Array, Sequence[float]]]
Alm = NDArray[np.complexfloating]


def number_from_cls(cls: Cls) -> int:
    """Return the number of fields from a list *cls*."""

    k = len(cls)
    n = int((2*k)**0.5)
    if n * (n + 1) // 2 != k:
        raise ValueError("length of cls is not a triangle number")
    return n


def cls_indices(n: int) -> Generator[Tuple[int, int], None, None]:
    """Iterate the indices *i*, *j* of *cls* for *n* fields."""
    for i in range(n):
        for j in range(i, -1, -1):
            yield (i, j)


def enumerate_cls(
    cls: Cls,
) -> Generator[Tuple[int, int, ArrayLike], None, None]:
    """Enumerate *cls*, returning a tuple *i, j, cl* for each *cl*."""

    n = number_from_cls(cls)
    for (i, j), cl in zip(cls_indices(n), cls):
        if cl is None:
            cl = []
        yield (i, j, cl)


def var_from_cl(cl):
    """Compute the field variance from an angular power spectrum."""

    ell = np.arange(np.shape(cl)[-1])
    return np.dot(2*ell + 1, cl)/(4*np.pi)


def iternorm(k: int, cov: Iterable[Array], size: Size = None
             ) -> Generator[Iternorm, None, None]:
    '''return the vector a and variance sigma^2 for iterative normal sampling'''

    n: Tuple[int, ...]
    if size is None:
        n = ()
    elif isinstance(size, int):
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
        if j is not None:
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


def cls2cov(cls: Cls, nl: int, nf: int, nc: int
            ) -> Generator[Array, None, None]:
    '''Return array of cls as a covariance matrix for iterative sampling.'''
    cov = np.zeros((nl, nc+1))
    end = 0
    for j in range(nf):
        begin, end = end, end + j + 1
        for i, cl in enumerate(cls[begin:end][:nc+1]):
            if cl is None:
                cov[:, i] = 0
            else:
                if i == 0 and np.any(np.less(cl, 0)):
                    raise ValueError('negative values in cl')
                n = len(cl)
                cov[:n, i] = cl
                cov[n:, i] = 0
        cov /= 2
        yield cov


def multalm(alm: Alm, bl: Array, inplace: bool = False) -> Alm:
    '''multiply alm by bl'''
    n = len(bl)
    if inplace:
        out = np.asanyarray(alm)
    else:
        out = np.copy(alm)
    for m in range(n):
        out[m*n-m*(m-1)//2:(m+1)*n-m*(m+1)//2] *= bl[m:]
    return out


def rescaled_alm(
    alm: Alm,
    cl: Array,
    fromcl: Optional[Array] = None,
    *,
    inplace: bool = False,
) -> Alm:
    """Scale *alm* by *cl*, optionally taking out *fromcl*."""
    fl: Array
    if fromcl is None:
        fl = cl
    else:
        # divide such that 0/0 -> 0
        fl = np.zeros(np.broadcast(cl, fromcl).shape)
        np.divide(cl, fromcl, where=(cl != 0), out=fl)
    # alms are scaled by the sqrt of (cl or cl/fromcl)
    alm = multalm(alm, np.sqrt(fl), inplace=inplace)
    # compute variance for alm and store in metadata
    var = var_from_cl(cl)
    update_metadata(alm, var=var)
    # done with updated alm
    return alm


def transform_cls(cls: Cls, tfm: ClTransform, pars: Tuple[Any, ...] = ()
                  ) -> Cls:
    '''Transform Cls to Gaussian Cls.'''
    gls = []
    for cl in cls:
        if cl is not None and len(cl) > 0:
            if cl[0] == 0:
                monopole = 0.
            else:
                monopole = None
            gl, info, err, niter = gaussiancl(cl, tfm, pars, monopole=monopole)
            if info == 0:
                warnings.warn('Gaussian cl did not converge, inexact transform')
        else:
            gl = []
        gls.append(gl)
    return gls


def discretized_cls(
    cls: Cls,
    *,
    lmax: Optional[int] = None,
    ncorr: Optional[int] = None,
    nside: Optional[int] = None,
) -> Cls:
    """Apply discretisation effects to angular power spectra.

    Depending on the given arguments, this truncates the angular power
    spectra to ``lmax``, removes all but ``ncorr`` correlations between
    fields, and applies the HEALPix pixel window function of the given
    ``nside``.  If no arguments are given, no action is performed.

    """

    if ncorr is not None:
        n = int((2*len(cls))**0.5)
        if n*(n+1)//2 != len(cls):
            raise ValueError("length of cls array is not a triangle number")
        cls = [cls[i*(i+1)//2+j] if j <= ncorr else [] for i in range(n) for j in range(i+1)]

    if nside is not None:
        pw = hp.pixwin(nside, lmax=lmax)

    gls = []
    for cl in cls:
        if cl is not None and len(cl) > 0:
            if lmax is not None:
                cl = cl[:lmax+1]
            if nside is not None:
                n = min(len(cl), len(pw))
                cl = cl[:n] * pw[:n]**2
        gls.append(cl)
    return gls


def biased_cls(
    cls: Cls,
    bias: ArrayLike,
) -> Cls:
    """Apply a linear bias factor to given *cls*."""

    # number of fields from cls array
    n = number_from_cls(cls)

    # bring given bias into shape with 1 number per field
    b = np.broadcast_to(bias, n)

    # multiply each cl by bias[i] and bias[j]
    return [np.multiply(b[i] * b[j], cl) for i, j, cl in enumerate_cls(cls)]


def lognormal_gls(cls: Cls, shift: float = 1) -> Cls:
    '''Compute Gaussian Cls for a lognormal random field.'''
    return transform_cls(cls, 'lognormal', (shift,))


def generate_alms(
    gls: Cls,
    *,
    ncorr: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> Generator[Alm, None, None]:
    """Iteratively sample Gaussian *alm* from Cls.

    A generator that iteratively samples the *alm* coefficients of
    Gaussian random fields with the given angular power spectra *gls*.

    The *gls* array must contain the auto-correlation of each new field
    followed by the cross-correlations with all previous fields in
    reverse order::

        gls = [
            gl_00,
            gl_11, gl_10,
            gl_22, gl_21, gl_20,
            ...
        ]

    Missing entries should be set to the empty list ``[]`` (but ``None``
    is also supported).

    The optional argument *ncorr* can be used to artificially limit how
    many realised *alm* arrays are correlated.  This saves memory, as
    only *ncorr* previous arrays need to be kept.

    """

    # get the default RNG if not given
    if rng is None:
        rng = np.random.default_rng()

    # number of fields
    ngrf = number_from_cls(gls)

    # number of correlated fields if not specified
    if ncorr is None:
        ncorr = ngrf - 1

    # number of modes
    n = max((len(gl) for gl in gls if gl is not None), default=0)
    if n == 0:
        raise ValueError("all gls are empty")

    # generates the covariance matrix for the iterative sampler
    cov = cls2cov(gls, n, ngrf, ncorr)

    # working arrays for the iterative sampling
    z = np.zeros(n*(n+1)//2, dtype=np.complex128)
    y = np.zeros((n*(n+1)//2, ncorr), dtype=np.complex128)

    # generate the conditional normal distribution for iterative sampling
    conditional_dist = iternorm(ncorr, cov, size=n)

    # sample the fields from the conditional distribution
    for it, (j, a, s) in enumerate(conditional_dist):
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
        alm[:n].imag[:] = 0

        # compute variance from the gl for this iteration
        var = var_from_cl(getcl(gls, it))

        # update metadata of alm with Gaussian random field properties
        update_metadata(alm, mean=0., var=var)

        # done with preparing the alm
        yield alm


def alm_to_gaussian(
    alm: Alm,
    nside: int,
    *,
    inplace: bool = False,
) -> Array:
    """Transform Gaussian *alm* coefficients to a Gaussian field."""

    # get alm metadata
    md = alm.dtype.metadata or {}

    # transform to map
    m = hp.alm2map(alm, nside, pixwin=False, pol=False, inplace=inplace)

    # set metadata of map
    update_metadata(m, **md)

    return m


def generate_gaussian(
    gls: Cls,
    nside: int,
    *,
    ncorr: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> Generator[Array, None, None]:
    """Iteratively sample Gaussian random fields from Cls.

    A generator that iteratively samples HEALPix maps of Gaussian random
    fields with the given angular power spectra ``gls`` and resolution
    parameter ``nside``.

    The optional argument ``ncorr`` can be used to artificially limit
    now many realised fields are correlated.  This saves memory, as only
    `ncorr` previous fields need to be kept.

    """

    # generate Gaussian alms and transform to maps in place
    for alm in generate_alms(gls, ncorr=ncorr, rng=rng):
        yield alm_to_gaussian(alm, nside, inplace=True)


def alm_to_lognormal(
    alm: Alm,
    nside: int,
    shift: float = 1.,
    var: Optional[float] = None,
    *,
    inplace: bool = False,
) -> Array:
    """Transform Gaussian *alm* coefficients to a lognormal field."""

    # compute the Gaussian random field
    m = alm_to_gaussian(alm, nside, inplace=inplace)

    # get variance from the metadata or the map itself if not provided
    if var is None:
        var = m.dtype.metadata and m.dtype.metadata.get("var") or np.var(m)

    # fix mean of the Gaussian random field for lognormal transformation
    m -= var/2

    # exponentiate values in place and subtract 1 in one operation
    np.expm1(m, out=m)

    # new variance for lognormal metadata
    var = np.expm1(var)

    # lognormal shift, unless unity
    if shift != 1:
        m *= shift
        var *= shift**2

    # update metadata of lognormal map
    update_metadata(m, mean=0., var=var)

    # done with the lognormal map
    return m


def generate_lognormal(
    gls: Cls,
    nside: int,
    shift: float = 1.,
    *,
    ncorr: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> Generator[Array, None, None]:
    """Iterative sample lognormal random fields from Gaussian Cls."""

    # generate the Gaussian alm and transform to lognormal in place
    for alm in generate_alms(gls, ncorr=ncorr, rng=rng):
        yield alm_to_lognormal(alm, nside, shift, inplace=True)


def getcl(cls, i, j=None, lmax=None):
    """Return a specific angular power spectrum from an array.

    Return the angular power spectrum for indices *i* and *j* from an
    array in *GLASS* ordering.

    Parameters
    ----------
    cls : list of array_like
        List of angular power spectra in *GLASS* ordering.
    i, j : int
        Combination of indices to return.  If *j* is not given, it is
        assumed equal to *i*.
    lmax : int, optional
        Truncate the returned spectrum at this mode number.

    Returns
    -------
    cl : array_like
        The angular power spectrum for indices *i* and *j*.

    """
    if j is None:
        j = i
    elif j > i:
        i, j = j, i
    cl = cls[i*(i+1)//2+i-j]
    if lmax is not None:
        cl = cl[:lmax+1]
    return cl
