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


def cov_reg_corr(cov, niter=20):
    '''covariance matrix from nearest correlation matrix

    Uses the algorithm of Higham (2000).

    '''

    # make sure shape is ok
    s = np.shape(cov)
    if len(s) < 2:
        raise TypeError('ndim(cov) < 2')
    if s[-2] != s[-1]:
        raise TypeError('cov is not square')

    # size of the covariance matrix
    n = s[-1]

    # make a copy to work on
    corr = np.copy(cov)

    # view onto the diagonal of the correlation matrix
    diag = corr.reshape(s[:-2] + (-1,))[..., ::n+1]

    # set correlations with nonpositive diagonal to zero
    good = (diag > 0)
    corr *= good[..., np.newaxis, :]
    corr *= good[..., :, np.newaxis]

    # get sqrt of the diagonal for normalization
    norm = np.sqrt(diag)

    # compute the correlation matrix
    np.divide(corr, norm[..., np.newaxis, :], where=good[..., np.newaxis, :], out=corr)
    np.divide(corr, norm[..., :, np.newaxis], where=good[..., :, np.newaxis], out=corr)

    # indices of the upper triangular part of the matrix
    triu = (...,) + np.triu_indices(n, 1)

    # always keep upper triangular part of matrix fixed to zero
    # otherwise, Dykstra's correction points in the wrong direction
    corr[triu] = 0

    # find the nearest covariance matrix with given diagonal
    dyks = np.zeros_like(corr)
    proj = np.empty_like(corr)
    for k in range(niter):
        # apply Dykstra's correction to current result
        np.subtract(corr, dyks, out=proj)

        # project onto positive semi-definite matrices
        w, v = np.linalg.eigh(proj)
        w[w < 0] = 0
        np.einsum('...ij,...j,...kj->...ik', v, w, v, out=corr)

        # keep upper triangular part fixed to zero
        corr[triu] = 0

        # compute Dykstra's correction
        np.subtract(corr, proj, out=dyks)

        # project onto matrices with unit diagonal
        diag[good] = 1

    # put the normalisation back to convert correlations to covariance
    np.multiply(corr, norm[..., np.newaxis, :], out=corr)
    np.multiply(corr, norm[..., :, np.newaxis], out=corr)

    # return the regularised covariance matrix
    return corr
