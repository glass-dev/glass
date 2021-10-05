# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''matter fields'''

__all__ = [
    'lognormal_matter',
]


from .types import Matter, Random
from .random import LognormalField


def lognormal_matter() -> Matter[Random]:
    '''matter field following a lognormal distribution'''

    return LognormalField(shift=1.0)
