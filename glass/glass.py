# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''meta-module for all known GLASS modules'''


def _glass_modules():
    '''return all GLASS modules'''

    from importlib import import_module
    from pkgutil import iter_modules

    ns = import_module('glass')
    modules = [name for _, name, _ in iter_modules(ns.__path__, 'glass.')]

    return {m[6:]: import_module(m) for m in modules if m != 'glass.glass'}


GLASS_MODULES = _glass_modules()

globals().update(GLASS_MODULES)

__all__ = list(GLASS_MODULES.keys())
