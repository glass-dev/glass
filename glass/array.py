# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''
Universal array support (:mod:`glass.array`)
============================================

.. currentmodule:: glass.array

The :mod:`glass.array` module provides support to write *GLASS* code that
supports a variety of compatible array types.



'''

from contextlib import contextmanager
from importlib import import_module
from typing import Any, Union

# these are underscored symbols which are to be exported from the namespace
_EXPORT = ['__array_api_version__', '__array_namespace__']

# placeholder type for an array namespace
ArrayNamespace = Any

# the currently used array namespace
_NS: Union[ArrayNamespace, None] = None


@contextmanager
def _restore(ns: Union[ArrayNamespace, None]) -> Any:
    '''context manager to restore a previous array namespace'''
    try:
        yield
    finally:
        use(ns)


def use(ns: Union[str, ArrayNamespace]) -> Any:
    '''Use the given array namespace.'''

    global _NS

    # store the previous namespace before it gets overwritten
    old_ns = _NS

    # set the new array namespace
    if isinstance(ns, str):
        # if the name of an array namespace was given, import it
        try:
            _NS = import_module(f'glass._array_api.{ns}')
        except ModuleNotFoundError as exc:
            raise ValueError(f'unknown array namespace "{ns}"') from exc
    else:
        # use the array namespace as given; could check type here
        _NS = ns

    # should check here that _NS really implements the protocol

    # we will modify the globals directly to change available symbols
    g = globals()

    # unload the current array namespace from module
    if old_ns is not None:
        for name in old_ns.__dict__:
            if name in _EXPORT or not name.startswith('_'):
                g.pop(name, None)

    # load the new array namespace into module
    for name, obj in _NS.__dict__.items():
        if name in _EXPORT or not name.startswith('_'):
            g[name] = obj

    # for use as a context manager
    return _restore(old_ns)


# use the numpy array namespace by default
use('numpy')
