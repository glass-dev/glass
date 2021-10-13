# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''matter fields'''

__glass__ = [
    'normal_matter',
    'lognormal_matter',
]


from .typing import NumberOfBins, RandomMatterFields
from .random_fields import NormalField, LognormalField


def normal_matter(nbins: NumberOfBins) -> RandomMatterFields:
    '''matter field following a normal distribution'''

    return [NormalField()]*nbins


def lognormal_matter(nbins: NumberOfBins) -> RandomMatterFields:
    '''matter field following a lognormal distribution'''

    return [LognormalField(shift=1.0)]*nbins
