# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''numerical methods'''

import numpy as np


def cov_reg_simple(cov):
    '''regularised covariance matrix by clipping negative eigenvalues'''

    # set negative eigenvalues to zero
    w, v = np.linalg.eigh(cov)
    w[w < 0] = 0

    # put matrix back together
    cov = np.einsum('...ij,...j,...kj->...ik', v, w, v)

    # fix the upper triangular part of the matrix to zero
    cov[(...,) + np.triu_indices(w.shape[-1], 1)] = 0

    # return the regularised covariance matrix
    return cov


def cov_reg_keepdiag(cov, niter=10):
    '''nearest covariance matrix with same diagonal

    Uses the algorithm of Higham (2000).

    '''

    # make sure shape is ok
    s = np.shape(cov)
    if len(s) < 2:
        raise TypeError('ndim(cov) < 2')
    if s[-2] != s[-1]:
        raise TypeError('cov is not square')

    # if matrix is positive semi-definite, no need for regularisation
        # no need for regularisation

    # size of the covariance matrix
    n = s[-1]

    # make a copy to work on
    _cov = np.copy(cov)

    # view onto the diagonal of the correlation matrix
    _dia = _cov.reshape(s[:-2] + (-1,))[..., ::n+1]

    # the fixed diagonal
    _fix = np.copy(_dia)

    # indices of the upper triangular part of the matrix
    _tri = (...,) + np.triu_indices(n, 1)

    # always keep upper triangular part of matrix fixed to zero
    # otherwise, Dykstra's correction points in the wrong direction
    _cov[_tri] = 0

    # find the nearest covariance matrix with given diagonal
    _cor = np.zeros_like(_cov)
    _pro = np.empty_like(_cov)
    for k in range(niter):
        # apply Dykstra's correction to current result
        np.subtract(_cov, _cor, out=_pro)

        # project onto positive semi-definite matrices
        w, v = np.linalg.eigh(_pro)
        w[w < 0] = 0
        np.einsum('...ij,...j,...kj->...ik', v, w, v, out=_cov)

        # keep upper triangular part fixed to zero
        _cov[_tri] = 0

        # compute Dykstra's correction
        np.subtract(_cov, _pro, out=_cor)

        # project onto matrices with given diagonal
        _dia[:] = _fix

    # return the regularised covariance matrix
    return _cov
