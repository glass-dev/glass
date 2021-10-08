# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''wrapper for cosmology'''

__all__ = [
    'cosmology',
]


from cosmology import LCDM
from .typing import Cosmology


def cosmology(**kwargs) -> Cosmology:
    '''construct a cosmology from keyword arguments'''
    return LCDM(**kwargs)
