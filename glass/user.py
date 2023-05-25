# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''
User utilities (:mod:`glass.user`)
==================================

.. currentmodule:: glass.user

The :mod:`glass.user` module contains convenience functions for users of the
library.


Input and output
----------------

.. autofunction:: save_cls
.. autofunction:: load_cls

'''

import numpy as np


def save_cls(filename, cls):
    '''Save a list of Cls to file.

    Uses :func:`numpy.savez` internally. The filename should therefore have a
    ``.npz`` suffix, or it will be given one.

    '''

    split = np.cumsum([len(cl) if cl is not None else 0 for cl in cls[:-1]])
    values = np.concatenate([cl for cl in cls if cl is not None])
    np.savez(filename, values=values, split=split)


def load_cls(filename):
    '''Load a list of Cls from file.

    Uses :func:`numpy.load` internally.

    '''

    with np.load(filename) as npz:
        values = npz['values']
        split = npz['split']
    return np.split(values, split)
