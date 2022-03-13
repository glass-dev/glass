# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''module for simulation control'''

import logging
import time
from datetime import timedelta
from collections.abc import Sequence, Mapping, Iterator, Iterable
import numpy as np

from .core import generator


log = logging.getLogger(__name__)


@generator('-> zmin, zmax')
def zgen(z):
    '''generator for contiguous redshift slices from a redshift array'''
    if isinstance(z, Iterable):
        z = iter(z)
    elif not isinstance(z, Iterator):
        raise TypeError('must be iterable or iterator')
    zmin, zmax = None, next(z)
    yield
    while True:
        try:
            zmin, zmax = zmax, next(z)
        except StopIteration:
            break
        log.info('zmin: %f', zmin)
        log.info('zmax: %f', zmax)
        yield zmin, zmax


@generator('-> zmin, zmax')
def zspace(zmin, zmax, *, dz=None, num=None):
    '''generator for redshift slices with uniform redshift spacing'''
    if (dz is None) == (num is None):
        raise ValueError('exactly one of "dz" or "num" must be given')
    if dz is not None:
        z = np.arange(zmin, zmax, dz)
    else:
        z = np.linspace(zmin, zmax, num+1)
    return zgen(z)


@generator('-> zmin, zmax')
def xspace(cosmo, zmin, zmax, *, dx=None, num=None):
    '''genrator for redshift slices with uniform comoving distance spacing'''
    if (dx is None) == (num is None):
        raise ValueError('exactly one of "dx" or "num" must be given')
    xmin, xmax = cosmo.dc(zmin), cosmo.dc(zmax)
    if dx is not None:
        x = np.arange(xmin, xmax, dx)
    else:
        x = np.linspace(xmin, xmax, num+1)
    z = cosmo.dc_inv(x)
    return zgen(z)


def _getitem_all(d, k):
    '''recursive dictionary getter'''
    if isinstance(k, str):
        if k.endswith('?'):
            return d.get(k[:-1], None)
        else:
            return d[k]
    elif isinstance(k, Sequence):
        return type(k)(_getitem_all(d, i) for i in k)
    elif isinstance(k, Mapping):
        return type(k)((i, _getitem_all(d, k[i])) for i in k)
    else:
        return d[k]


def _setitem_all(d, k, v):
    '''recursive dictionary setter'''
    if isinstance(k, str):
        d[k] = v
    elif isinstance(k, Sequence):
        if not isinstance(v, Sequence):
            raise TypeError('cannot set sequence of keys with non-sequence of values')
        if len(k) != len(v):
            raise ValueError(f'cannot set {len(k)} items with {len(v)} values')
        for k_, v_ in zip(k, v):
            _setitem_all(d, k_, v_)
    elif isinstance(k, Mapping):
        if not isinstance(v, Mapping):
            raise TypeError('cannot set mapping of keys with non-mapping of values')
        for i in k:
            _setitem_all(d, k[i], v[i])
    else:
        d[k] = v


def generate(generators):
    '''run generators'''

    log.info('=== initialize ===')

    # prime all generators
    for g in generators:
        log.info('--- %s ---', g.name)
        log.debug('initial signature: %s', g.signature)
        g.send(None)
        log.debug('final signature: %s', g.signature)

    # this will keep the state of the simulation during iteration
    state = {}

    # simulation status
    n = 0
    t0 = time.monotonic()

    # iterate simulation
    # for each iteration, run all generators, then yield the results
    # for each generator, collect its inputs and store its outputs
    # stop if any of the generators stops by throwing StopIteration
    while True:
        n += 1

        state['#'] = n

        log.info('=== shell %d ===', n)

        for g in generators:

            t = time.monotonic()

            log.info('--- %s ---', g.name)

            if g._inputs is not None:
                inputs = _getitem_all(state, g._inputs)
            else:
                inputs = None

            try:
                values = g.send(inputs)
            except StopIteration:
                log.info('>>> generator has stopped the simulation <<<')
                break

            if g._outputs is not None:
                _setitem_all(state, g._outputs, values)

            log.info('>>> %s: %s <<<', g.name, timedelta(seconds=time.monotonic()-t))

        else:  # no break
            yield state
            continue

        # break in inner loop
        break

    log.info('=== finalize ===')

    # close all generators
    for g in generators:
        log.info('--- %s ---', g.name)
        g.close()

    log.info('>>> done in %s <<<', timedelta(seconds=time.monotonic()-t0))
