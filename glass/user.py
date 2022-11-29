# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''module for user utilities'''

import numpy as np

from .matter import MatterWeights
from .lensing import LensingWeights


def save_shells(filename, shells, mweights=None, cls=None, lweights=None):
    '''Save shell definitions to file.'''
    kws = {'shells': shells}
    if mweights is not None:
        kws['mweights'] = mweights
    if cls is not None:
        kws['cls'] = cls
    if lweights is not None:
        kws['lweights'] = np.insert(lweights.w, 0, lweights.z, 1)

    np.savez(filename, **kws)


def load_shells(filename):
    '''Load shell definitions from file.'''
    with np.load(filename) as npz:
        shells = npz['shells']
        mweights = npz.get('mweights', None)
        cls = npz.get('cls', None)
        lweights = npz.get('lweights', None)
    if mweights is not None:
        mweights = MatterWeights._make(mweights)
    if lweights is not None:
        lweights = LensingWeights._make((lweights[..., 0], lweights[..., 1:]))

    return shells, mweights, cls, lweights
