# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''module for user utilities'''

import numpy as np

from .matter import MatterWeights


def save_shells(filename, shells, mweights=None, cls=None):
    '''Save shell definitions to file.'''
    kws = {'shells': shells}
    if mweights is not None:
        kws['mweights'] = mweights
    if cls is not None:
        kws['cls'] = cls

    np.savez(filename, **kws)


def load_shells(filename):
    '''Load shell definitions from file.'''
    with np.load(filename) as npz:
        shells = npz['shells']
        mweights = npz.get('mweights', None)
        cls = npz.get('cls', None)
    if mweights is not None:
        mweights = MatterWeights._make(mweights)

    return shells, mweights, cls
