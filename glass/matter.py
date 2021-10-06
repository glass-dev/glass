# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''matter fields'''

__all__ = [
    'normal_matter',
    'lognormal_matter',
]


from .types import NumberOfBins, Matter, Random
from .random import NormalField, LognormalField


def normal_matter(nbins: NumberOfBins) -> Matter[Random]:
    '''matter field following a normal distribution'''

    return [NormalField()]*nbins


def lognormal_matter(nbins: NumberOfBins) -> Matter[Random]:
    '''matter field following a lognormal distribution'''

    return [LognormalField(shift=1.0)]*nbins
