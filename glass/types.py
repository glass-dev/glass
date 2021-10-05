# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''common types'''

from typing import Any, Annotated, get_args, get_origin
from numpy.typing import ArrayLike


# simulation
RedshiftBins = Annotated[list[float], 'zbins']
NumberOfBins = Annotated[int, 'nbins']
Cosmology = Annotated[Any, 'cosmology']
ClsDict = Annotated[dict[tuple[str, str], ArrayLike], 'cls']

# fields
MatterField = Annotated[ArrayLike, 'matter']
ConvergenceField = Annotated[ArrayLike, 'convergence']
ShearField = Annotated[ArrayLike, 'shear']


def get_default_ref(T):
    '''return the default ref from an annotated type'''

    if get_origin(T) == Annotated:
        args = get_args(T)
        if len(args) > 1:
            return args[1]

    return None
