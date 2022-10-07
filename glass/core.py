# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''module for core functionality'''

import logging
import time
import os.path
import pickle
from functools import wraps
from datetime import timedelta
from collections import UserDict
from collections.abc import Sequence, Mapping

from .generator import generator


log = logging.getLogger(__name__)


# variable definitions
ITER = 'iteration'
'''The number of the current iteration, starting from 1.'''


class GeneratorError(RuntimeError):
    '''raised when an error occurred in a generator'''

    def __init__(self, generator, state):
        '''construct a GeneratorError for generator and state'''
        self._generator = generator
        self._state = state
        g, n = generator.__name__, state[ITER]
        super().__init__(f'iteration {n}: uncaught exception in {g}')

    @property
    def generator(self):
        '''the generator that caused the exception'''
        return self._generator

    @property
    def state(self):
        '''the state of the simulation when the exception occurred'''
        return self._state


class State(UserDict):
    '''simulation state with recursive lookups and context'''

    Nothing = object()

    def __init__(self, data=None, context=None, **kwargs):
        super().__init__(data, **kwargs)
        self.context = context
        self.data['state'] = self

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


def _gencall(generator, state):
    '''call a generator'''

    t = time.monotonic()

    log.info('--- %s ---', generator.__name__)

    if state is GeneratorExit:
        generator.close()
    elif state is None:
        generator.send(None)
    else:
        receives = getattr(generator, 'receives', None)
        yields = getattr(generator, 'yields', None)
        state[yields] = generator.send(state[receives])

    log.info('>>> %s: %s <<<', generator.__name__, timedelta(seconds=time.monotonic()-t))


def generate(generators):
    '''run generators

    Raises
    ------
    :class:`GeneratorError`
        If an uncaught exception is raised by a generator.  The original
        exception is the ``__cause__`` of the :class:`GeneratorError`.

    '''

    log.info('=== initialize ===')

    # prime all generators
    for g in generators:
        _gencall(g, None)

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

        state = State()
        state[ITER] = n

        log.info('=== iteration %d ===', n)

        for g in generators:
            try:
                _gencall(g, state)
            except StopIteration:
                log.info('>>> generator has stopped the simulation <<<')
                break
            except Exception as e:
                raise GeneratorError(g, state) from e
        else:  # no break
            ty = time.monotonic()
            log.info('--- yield ---')
            yield state
            log.info('>>> yield: %s <<<', timedelta(seconds=time.monotonic()-ty))
            log.info('»»» iteration %d: %s «««', n, timedelta(seconds=time.monotonic()-ts))
            continue
        break

    log.info('=== finalize ===')

    # close all generators
    for g in generators:
        _gencall(g, GeneratorExit)

    log.info('>>> done in %s <<<', timedelta(seconds=time.monotonic()-t0))


def run(generators):
    '''Run all generators without yielding.

    This function runs the given generators without yielding the results of each
    iteration.

    Parameters
    ----------
    generators : list of generator
        The generators to run.

    Notes
    -----
    Calling :func:`run` is functionally equivalent to calling :func:`generate`
    in a trivial loop::

        for _ in generate(generators):
            pass

    '''
    for _ in generate(generators):
        pass


def group(name, generators):
    '''group generators under a common name'''

    # create a generator with the named output
    @generator(receives='state', yields=name)
    def g():
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

            for g in generators:
                _gencall(g, state)

        # finalise sub-generators
        for g in generators:
            _gencall(g, GeneratorExit)

    # also update the name of the group for printing
    g.__name__ = f'group "{name}"'

    # return the generator just created
    return g()


def save(filename, variables):
    '''Save variables to file on each iteration.

    This generator receives the nominated variables on each iteration and saves
    their values to ``filename``.  The variables can subsequently be read from
    the file and iterated using the :func:`load` generator.

    Parameters
    ----------
    filename : str
        Filename for the stored variables.  If ``filename`` does not end in
        ``'.glass'``, the suffix will be appended.
    variables : list of str
        The variables to be saved.

    Yields
    ------
    ---

    Receives
    --------
    (variables)
        All nominated variables.

    Warnings
    --------
    No checking at all is perfomed on the given list of variables.  Make sure
    that it contains everything that may be required for a subsequent run.

    '''

    # make filename with '.glass' extension
    root, ext = os.path.splitext(filename)
    if ext != '.glass':
        filename = root + ext + '.glass'

    # construct generator which receives ITER and all variables
    @generator(receives=(*variables,))
    @wraps(save)
    def g():
        log.info('filename: %s', filename)
        log.info('variables:')
        for v in variables:
            log.info('- %s', v)
        with open(filename, 'wb') as f:
            pickle.dump(variables, f)
            while True:
                values = yield
                pickle.dump(values, f)

    return g()


def load(filename):
    '''Load variables from file on each iteration.

    This generator reads the saved variables from ``filename`` and yields their
    values on each iteration.  The file would normally be created with the
    :func:`save` generator.

    Parameters
    ----------
    filename : str
        Filename for the stored variables.  If ``filename`` does not end in
        ``'.glass'``, the suffix will be appended.

    Yields
    ------
    (variables)
        All saved variables.

    Receives
    --------
    ---

    Warnings
    --------
    To allow loading of any kind of data, this function uses the :mod:`pickle`
    module, which can in principle be used to execute arbitrary code on loading.
    Therefore, you should only load data that you trust.

    '''

    # make filename with '.glass' extension
    root, ext = os.path.splitext(filename)
    if ext != '.glass':
        filename = root + ext + '.glass'

    # read variables from archive
    with open(filename, 'rb') as f:
        variables = pickle.load(f)

    # construct generator which yields all variables
    @generator(yields=(*variables,))
    @wraps(load)
    def g():
        log.info('filename: %s', filename)
        with open(filename, 'rb') as f:
            variables = pickle.load(f)
            log.info('variables:')
            for v in variables:
                log.info('- %s', v)
            values = None
            while True:
                yield values
                try:
                    values = pickle.load(f)
                except EOFError:
                    break

    return g()
