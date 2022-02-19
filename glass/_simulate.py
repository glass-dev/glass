# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''internal module for simulation control'''

import logging
import time
from datetime import timedelta
from collections.abc import Sequence, Mapping


log = logging.getLogger('glass')


def _getitem_all(d, k):
    '''recursive dictionary getter'''
    if isinstance(k, str):
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


def simulate(zbins, generators):
    '''simulate a light cone'''

    log.info('=== initialize ===')

    # prime all generators
    for g in generators:
        log.info('--- %s ---', g.name)
        log.debug('signature: %s', g.signature)
        g.send(None)

    # this will keep the state of the simulation during iteration
    state = {}

    # loop over redshift bins
    for zmin, zmax in zip(zbins, zbins[1:]):

        log.info('=== %f to %f ===', zmin, zmax)

        # redshift bounds for this slice
        state['zmin'] = zmin
        state['zmax'] = zmax

        # run all generators for this redshift slice
        for g in generators:

            t = time.monotonic()

            name = getattr(g, '__name__', '<anonymous>')

            log.info('--- %s ---', name)

            if g._inputs is not None:
                inputs = _getitem_all(state, g._inputs)
            else:
                inputs = None

            values = g.send(inputs)

            if g._outputs is not None:
                _setitem_all(state, g._outputs, values)

            log.info('done in %s', timedelta(seconds=time.monotonic()-t))

    log.info('=== finalize ===')

    # close all generators
    for g in generators:
        log.info('--- %s ---', g.name)
        g.close()
