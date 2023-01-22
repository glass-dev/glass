# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''
User utilities (:mod:`glass.user`)
==================================

.. currentmodule:: glass.user

The :mod:`glass.user` module contains convenience functions for users of the
library.

.. autosummary::
   :template: generator.rst
   :toctree: generated/
   :nosignatures:

    Profiler
    profile

'''

import logging
import time
import tracemalloc
from datetime import timedelta
from contextlib import contextmanager

logger = logging.getLogger(__name__)


def _memf(n: int):
    '''Format a number of bytes in human-readable form.'''
    n = float(n)
    for x in ['', 'k', 'M', 'G', 'T']:
        if n < 1000:
            break
        n /= 1000
    return f'{n:.2f}{x}'


class Profiler:
    '''Simple procedural profiling.'''

    DEFAULT_LOGGER: logging.Logger = logger
    '''The default logger to use for instances of this class.'''

    def __init__(self, logger: logging.Logger = None):
        '''Create a new instance of a profiler.'''
        self._logger = logger if logger is not None else self.DEFAULT_LOGGER
        self._time = {}

    def start(self, message: str = None):
        '''Start profiler.'''
        self.log(message, 'start')

    def stop(self, message: str = None):
        '''Stop profiler.'''
        self.log(message, 'start')
        self._time = {}

    def log(self, message: str = None, timer: str = None):
        '''Log a profiling point.'''
        t0 = self._time.get(timer)
        t1 = self._time[timer] = time.monotonic()
        parts = []
        if message is not None:
            parts.append(message)
        if t0 is not None:
            parts.append(f'time {timedelta(seconds=t1-t0)}')
        if tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            parts.append(f'mem {_memf(current)}')
            parts.append(f'peak {_memf(peak)}')
        msg = ' -- '.join(parts)
        self._logger.info('### %s', msg)

    def loop(self, message: str = None):
        '''Log a profiling point for a loop.'''
        self.log(message, 'loop')


@contextmanager
def profile(message: str = None, logger: logging.Logger = None):
    '''Context manager for simple profiling.'''
    prof = Profiler(logger)
    try:
        prof.start(message)
        yield prof
    finally:
        prof.stop(message)
