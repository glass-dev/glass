# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''module for simulation control'''

import logging
import time
from datetime import timedelta
from collections import UserDict
from collections.abc import Sequence, Mapping, Iterator, Iterable
import numpy as np

from .generator import self, yields


log = logging.getLogger(__name__)


class GeneratorError(RuntimeError):
    '''raised when an error occurred in a generator'''

    def __init__(self, generator, state):
        '''construct a GeneratorError for generator and shell'''
        self._generator = generator
        self._state = state
        g, n = generator.__name__, state['#']
        super().__init__(f'shell {n}: uncaught exception in {g}')

    @property
    def generator(self):
        '''the generator that caused the exception'''
        return self._generator

    @property
    def state(self):
        '''the state of the simulation when the exception occurred'''
        return self._state


@yields('zmin', 'zmax')
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


@yields('zmin', 'zmax')
def zspace(zmin, zmax, *, dz=None, num=None):
    '''generator for redshift slices with uniform redshift spacing'''
    if (dz is None) == (num is None):
        raise ValueError('exactly one of "dz" or "num" must be given')
    if dz is not None:
        z = np.arange(zmin, np.nextafter(zmax+dz, zmax), dz)
    else:
        z = np.linspace(zmin, zmax, num+1)
    yield from zgen(z)


@yields('zmin', 'zmax')
def xspace(cosmo, zmin, zmax, *, dx=None, num=None):
    '''genrator for redshift slices with uniform comoving distance spacing'''
    if (dx is None) == (num is None):
        raise ValueError('exactly one of "dx" or "num" must be given')
    xmin, xmax = cosmo.dc(zmin), cosmo.dc(zmax)
    if dx is not None:
        x = np.arange(xmin, np.nextafter(xmax+dx, xmax), dx)
    else:
        x = np.linspace(xmin, xmax, num+1)
    z = cosmo.dc_inv(x)
    yield from zgen(z)


class State(UserDict):
    '''simulation state with recursive lookups and context'''

    Nothing = object()

    def __init__(self, data=None, context=None, **kwargs):
        super().__init__(data, **kwargs)
        self.context = context

    def __setitem__(self, key, value):
        self.setall(key, value)

    def __getitem__(self, key):
        return self.getall(key)

    def getcontext(self, key, default=Nothing):
        '''return item from self or context'''
        if key in self.data:
            return self.data[key]
        if self.context is not None:
            try:
                return self.context[key]
            except KeyError:
                pass
        if default is not State.Nothing:
            return default
        raise KeyError(key)

    def getall(self, key):
        '''recursive dictionary getter'''
        if key is None:
            return None
        elif isinstance(key, str):
            if key.endswith('?'):
                return self.getcontext(key[:-1], None)
            else:
                return self.getcontext(key)
        elif isinstance(key, Sequence):
            return type(key)(self.getall(k) for k in key)
        elif isinstance(key, Mapping):
            return type(key)((k, self.getall(key[k])) for k in key)
        else:
            return self.getcontext(key)

    def setall(self, key, value):
        '''recursive dictionary setter'''
        if key is None:
            pass
        elif isinstance(key, str):
            self.data[key] = value
        elif isinstance(key, Sequence):
            if not isinstance(value, Sequence):
                raise TypeError('cannot set sequence of keys with non-sequence of values')
            if len(key) != len(value):
                raise ValueError(f'cannot set {len(key)} items with {len(value)} values')
            for k, v in zip(key, value):
                self.setall(k, v)
        elif isinstance(key, Mapping):
            if not isinstance(value, Mapping):
                raise TypeError('cannot set mapping of keys with non-mapping of values')
            for k in key:
                self.setall(key[k], value[k])
        else:
            self.data[key] = value


def _gencall(generator, state, initial=False):
    '''call a generator'''

    t = time.monotonic()

    log.info('--- %s ---', generator.__name__)

    if state is not None:
        if initial:
            receives = None
            yields = getattr(generator, 'initial', None)
        else:
            receives = getattr(generator, 'receives', None)
            yields = getattr(generator, 'yields', None)
        state[yields] = generator.send(state[receives])
    else:
        generator.close()

    log.info('>>> %s: %s <<<', generator.__name__, timedelta(seconds=time.monotonic()-t))


def generate(generators):
    '''run generators

    Raises
    ------
    GeneratorError
        If an uncaught exception is raised by a generator.  The original
        exception is the `__cause__` of the GeneratorError.

    '''

    log.info('=== initialize ===')

    # this will keep the state of the simulation during iteration
    state = State()
    state['state'] = state

    # prime all generators
    for g in generators:
        _gencall(g, state, initial=True)

    # simulation status
    n = 0
    t0 = time.monotonic()

    # iterate simulation
    # for each iteration, run all generators, then yield the results
    # for each generator, collect its inputs and store its outputs
    # stop if any of the generators stops by throwing StopIteration
    # raise a GeneratorError if there is an uncaught exception
    while True:
        n += 1
        ts = time.monotonic()

        state['#'] = n

        log.info('=== shell %d ===', n)

        for g in generators:
            try:
                _gencall(g, state)
            except StopIteration:
                log.info('>>> generator has stopped the simulation <<<')
                break
            except BaseException as e:
                raise GeneratorError(g, state) from e
        else:  # no break
            ty = time.monotonic()
            log.info('--- yield ---')
            yield state
            log.info('>>> yield: %s <<<', timedelta(seconds=time.monotonic()-ty))
            log.info('»»» shell %d: %s «««', n, timedelta(seconds=time.monotonic()-ts))
            continue
        break

    log.info('=== finalize ===')

    # close all generators
    for g in generators:
        _gencall(g, None)

    log.info('>>> done in %s <<<', timedelta(seconds=time.monotonic()-t0))


@self
def group(self, name, generators):
    '''group generators under a common name'''

    # set the generator inputs and outputs
    self.receives = 'state'
    self.yields = name

    # also update the name of the group for printing
    self.__name__ += f' "{name}"'

    # prime sub-generators
    for g in generators:
        _gencall(g, None)

    # initial yield
    state = None

    # on every iteration, store sub-generators output in sub-state
    while True:
        try:
            context = yield state
        except GeneratorExit:
            break

        state = State(context=context)
        state['state'] = state
        for g in generators:
            _gencall(g, state)

    # finalise sub-generators
    for g in generators:
        _gencall(g, GeneratorExit)
