# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''random fields'''

__all__ = [
    'collect_cls',
    'transform_gaussian_cls',
    'regularize_gaussian_cls',
    'transform_regularized_cls',
    'gaussian_random_fields',
    'transform_random_fields',
    'generate_random_fields',
]


import numpy as np
import healpy as hp
import logging
from sortcl import cl_indices, enumerate_cls

from .numeric import cov_reg_simple, cov_reg_corr


log = logging.getLogger('glass.random')


def _num_fields_from_num_cls(m):
    '''get number of fields from number of cls'''
    n = int((2*m)**0.5)
    if m != n*(n+1)//2:
        raise TypeError(f'number of cls is not a triangle number: {m}')
    return n


def collect_cls(cls_dict, names, *, allow_missing=False):
    '''collect cls for a list of random fields'''

    # number of fields
    n = len(names)

    log.info('collecting cls for %d random fields...', n)
    for _ in names:
        log.debug('- %s', _)

    cls = []
    for i, j in zip(*cl_indices(n)):
        a, b = names[i], names[j]

        if (a, b) in cls_dict:
            cl = cls_dict[a, b]
        elif (b, a) in cls_dict:
            cl = cls_dict[b, a]
        elif allow_missing:
            cl = None
            log.warning('WARNING: missing cl: %s-%s', a, b)
        else:
            raise KeyError(f'missing cls: {a}-{b}')

        cls.append(cl)

    log.info('collected %d cls, with %d missing', len(cls), sum(_ is None for _ in cls))

    return cls


def cls_as_dict(cls, names):
    '''sort cls into a dictionary given names'''
    return {(names[i], names[j]): cl for i, j, cl in enumerate_cls(cls) if cl is not None}


def transform_gaussian_cls(cls, fields, nside=None):
    '''transform cls to Gaussian cls for simulation'''

    # number of fields must match number of cls
    n = len(fields)
    if len(cls) != n*(n+1)//2:
        raise TypeError(f'requires {n*(n+1)//2} cls for {n} random fields')

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
    for i, j, cl in enumerate_cls(cls):
        # only work on available cls
        if cl is not None:
            # simulating integrated maps by multiplying cls and pw
            # shorter array determines length
            cl_len = min(len(cl), lmax+1)
            cl = cl[:cl_len]*pw[:cl_len]

            # autocorrelation must be nonnegative
            if i == j:
                neg_cl = np.where(cl < 0)[0]
                if len(neg_cl) > 0:
                    log.warn('WARNING: negative cls for field %d', i)
                    log.debug('negative cls at l = %s', neg_cl)

            # transform the cl
            cl = (fields[i] & fields[j])(cl)

            # Gaussian autocorrelation must be nonnegative
            if i == j:
                neg_cl = np.where(cl < 0)[0]
                if len(neg_cl) > 0:
                    log.warn('WARNING: negative Gaussian cls for field %d', i)
                    log.debug('negative cls at l = %s', neg_cl)

        # store the Gaussian cl, or None
        gaussian_cls.append(cl)

    # returns the list of transformed cls in input order
    return gaussian_cls


def regularize_gaussian_cls(cls, method='corr'):
    '''regularize Gaussian cls for random sampling'''

    # debug output computations are expensive, so only do them when necessary
    debug = log.isEnabledFor(logging.DEBUG)

    # number of fields
    n = _num_fields_from_num_cls(len(cls))

    # maximum l in input cls
    lmax = np.max([len(cl)-1 for cl in cls if cl is not None])

    log.debug('create covariance matrix...')

    # this is the covariance matrix of cls
    # the leading dimension is l, then it is a n x n covariance matrix
    # missing entries are zero, which is the default value
    cov = np.zeros((lmax+1, n, n))

    # fill the matrix up by going through the cls in order
    # if the cls list is ragged, some entries at high l may remain zero
    # only fill the lower triangular part, everything is symmetric
    for i, j, cl in enumerate_cls(cls):
        if cl is not None:
            cov[:len(cl), j, i] = cl

    log.debug('check covariance matrix...')

    # use cholesky() as a fast way to check for positive semi-definite
    try:
        np.linalg.cholesky(cov + np.finfo(0.).tiny)
    except np.linalg.LinAlgError:
        # matrix needs regularisation
        pass
    else:
        # matrix is ok
        return cls

    log.info('cls require regularisation!')

    log.debug('covariance matrix regularisation...')
    log.debug('regularisation method: %s', method)

    if method == 'simple':
        reg = cov_reg_simple(cov)
    elif method == 'corr':
        reg = cov_reg_corr(cov)
    else:
        raise ValueError(f'unknown method "{method}" for regularisation')

    # show the maximum change in each l
    if debug:
        # get the diagonal of covariance matrix and regularised
        _diag_cov = cov.reshape(lmax+1, -1)[:, ::n+1]
        _diag_reg = reg.reshape(lmax+1, -1)[:, ::n+1]

        _abs = np.fabs(_diag_reg - _diag_cov)
        _rel = np.divide(_abs, np.fabs(_diag_cov), where=(_diag_cov != 0), out=np.zeros_like(_abs))
        _max_abs = np.max(_abs, axis=-1)
        _max_rel = np.max(_rel, axis=-1)
        _argmax_abs = np.argmax(_max_abs)
        _argmax_rel = np.argmax(_max_rel)
        with np.printoptions(precision=3, linewidth=np.inf, floatmode='fixed', sign=' '):
            log.debug('maximum absolute change in autocorrelation: %g [l = %d]', _max_abs[_argmax_abs], _argmax_abs)
            log.debug('before:')
            log.debug('%s', cov[_argmax_abs].diagonal())
            log.debug('after:')
            log.debug('%s', reg[_argmax_abs].diagonal())
            log.debug('maximum relative change in autocorrelation: %g [l = %d]', _max_rel[_argmax_rel], _argmax_rel)
            log.debug('before:')
            log.debug('%s', cov[_argmax_rel].diagonal())
            log.debug('after:')
            log.debug('%s', reg[_argmax_rel].diagonal())

    # gather regularised Gaussian cls from array
    reg_gaussian_cls = []
    for i, j in zip(*cl_indices(n)):
        # convert this to a contiguous array as its passed to C healpix
        cl = np.ascontiguousarray(reg[:, j, i])
        reg_gaussian_cls.append(cl)

    # return the regularised Gaussian cls
    return reg_gaussian_cls


def transform_regularized_cls(cls, fields):
    '''transform regularized Gaussian Cls to regularized Cls'''

    # number of fields must match number of cls
    n = len(fields)
    if len(cls) != n*(n+1)//2:
        raise TypeError(f'requires {n*(n+1)//2} cls for {n} random fields')

    # apply the inverse transforms to the regularized Gaussian cls
    regularized_cls = []
    for i, j, cl in enumerate_cls(cls):
        if cl is not None:
            cl = (fields[i] & fields[j])(cl, inv=True)
        regularized_cls.append(cl)

    # returns the list of transformed cls in input order
    return regularized_cls


def gaussian_random_fields(nside, cls):
    '''sample Gaussian random fields from cls'''

    # debug output computations are expensive, so only do them when necessary
    debug = log.isEnabledFor(logging.DEBUG)

    log.debug('sampling alms...')

    # sample the Gaussian random fields in harmonic space
    alms = hp.synalm(cls, new=True)

    if debug:
        lmax = hp.Alm.getlmax(alms[0].size)
        l = np.arange(lmax+1)
        two_l_plus_1_over_4_pi = (2*l+1)/(4*np.pi)
        for i, alm in enumerate(alms):
            _cl = hp.alm2cl(alm)
            _mean = np.sqrt(4*np.pi*_cl[0])
            _var = two_l_plus_1_over_4_pi@_cl
            _theory_var = two_l_plus_1_over_4_pi@cls[i]
            log.debug('- mean: %g [%g]', _mean, 0.)
            log.debug('  var: %g [%g]', _var, _theory_var)

    log.debug('computing maps...')

    # compute the Gaussian random field maps in real space
    # can be performed in place because the alms are not needed
    maps = hp.alm2map(alms, nside, pixwin=False, pol=False, inplace=True)
    alms = None

    if debug:
        for i, m in enumerate(maps):
            _theory_var = two_l_plus_1_over_4_pi@cls[i]
            log.debug('- mean: %g [%g]', np.mean(m), 0.)
            log.debug('  var: %g [%g]', np.var(m), _theory_var)
            log.debug('  min: %g', np.min(m))
            log.debug('  max: %g', np.max(m))

    # fields have been created
    return maps


def transform_random_fields(maps, fields, cls):
    '''apply transformations to Gaussian random fields'''

    # number of fields must match number of cls
    n = len(fields)
    if len(cls) != n*(n+1)//2:
        raise TypeError(f'requires {n*(n+1)//2} cls for {n} random fields')

    log.debug('computing variances...')

    # compute the variances of the Gaussian fields from cls
    # these may be used by the transforms
    var = []
    for i, j, cl in enumerate_cls(cls):
        if i == j:
            l = np.arange(len(cl))
            v = np.sum((2*l+1)/(4*np.pi)*cl)
            var.append(v)

    log.debug('transforming fields...')

    # transform the Gaussian random fields to the output fields
    for i, field in enumerate(fields):
        log.debug('- field %d: %s', i, field)

        maps[i] = field(maps[i], var=var[i])

    # maps have been changed in place, return nonetheless
    return maps


def generate_random_fields(nside, cls, random_fields, *, allow_missing_cls=False,
                           regularization='corr', return_cls=False):
    '''generate random fields from cls'''

    # collect the names of all random fields
    names = [f'{name}[{i}]' for name, _ in random_fields.items() for i in range(len(_))]

    # make a flat list of random fields
    fields = sum(random_fields.values(), [])

    # collect the flat list of cls for the random fields
    cls = collect_cls(cls, names, allow_missing=allow_missing_cls)

    # compute the Gaussian cls from the intended output cls
    gaussian_cls = transform_gaussian_cls(cls, fields, nside)

    # regularize the Gaussian cls and transform the regularized cls back
    if regularization is not False:
        reg_gaussian_cls = regularize_gaussian_cls(gaussian_cls, method=regularization)
        regularized_cls = transform_regularized_cls(reg_gaussian_cls, fields)
    else:
        reg_gaussian_cls = gaussian_cls
        regularized_cls = cls

    # compute the Gaussian random fields
    maps = gaussian_random_fields(nside, reg_gaussian_cls)

    # transform to the output fields
    maps = transform_random_fields(maps, fields, cls)

    # assign flat list of maps to fields
    out_fields = {}
    map_offset = 0
    for name, _fields in random_fields.items():
        length = len(_fields)
        out_fields[name] = maps[map_offset:map_offset+length]
        map_offset += length

    # keep track of what this function returns
    out = out_fields

    # also store the cls used if asked to
    if return_cls:
        out_cls = {
            'cls': cls_as_dict(cls, names),
            'gaussian_cls': cls_as_dict(gaussian_cls, names),
            'reg_gaussian_cls': cls_as_dict(reg_gaussian_cls, names),
            'regularized_cls': cls_as_dict(regularized_cls, names),
        }
        out = out, out_cls

    log.debug('random fields generated')

    return out
