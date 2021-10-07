# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''type hinting'''

__all__ = [
    'RedshiftBins',
    'NumberOfBins',
    'Cosmology',
    'ClsDict',
    'Random',
    'ClsList',
    'GaussianClsList',
    'RegGaussianClsList',
    'Matter',
    'Convergence',
    'Shear',
    'get_annotation',
]


from typing import TypeVar, Any, Annotated, NamedTuple, get_args, get_origin
from numpy.typing import ArrayLike

T = TypeVar('T')

# simulation
NSide = Annotated[int, 'name:nside']
RedshiftBins = Annotated[list[float], 'name:zbins']
NumberOfBins = Annotated[int, 'name:nbins']
Cosmology = Annotated[Any, 'name:cosmology']
ClsDict = Annotated[dict[tuple[str, str], ArrayLike], 'name:cls']

# random fields
Random = Annotated[list['RandomField'], 'random']
ClsList = list[ArrayLike]
GaussianClsList = list[ArrayLike]
RegGaussianClsList = list[ArrayLike]

# fields
Matter = Annotated[T, 'name:matter']
Convergence = Annotated[T, 'name:convergence']
Shear = Annotated[T, 'name:shear']


class _Annotation(NamedTuple):
    '''extra information from annotated type hints'''

    name: str
    random: bool


def get_annotation(T):
    '''return the extra information from annotated type hints'''

    name = None
    random = False

    if get_origin(T) == Annotated:
        args = get_args(T)
        for arg in args[1:]:
            if isinstance(arg, str):
                if arg.startswith('name:'):
                    name = arg[5:]
                elif arg == 'random':
                    random = True

    return _Annotation(name=name, random=random)
