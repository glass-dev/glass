# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''type hinting'''

__all__ = [
    'WorkDir',
    'NSide',
    'LMax',
    'RedshiftBins',
    'NumberOfBins',
    'Cosmology',
    'Cls',
    'TheoryCls',
    'SampleCls',
    'RandomFields',
    'RandomMatterFields',
    'RandomConvergenceFields',
    'Fields',
    'MatterFields',
    'ConvergenceFields',
    'ShearFields',
    'GalaxyFields',
    'get_annotation',
    'annotate',
]


from functools import wraps
from typing import TypeVar, Any, Annotated, Union, get_args, get_origin
from numpy.typing import ArrayLike

T = TypeVar('T')
NoneType = type(None)

# simulation
WorkDir = Annotated[str, 'glass:workdir']
NSide = Annotated[int, 'glass:nside']
LMax = Annotated[int, 'glass:lmax']
RedshiftBins = Annotated[list[float], 'glass:zbins']
NumberOfBins = Annotated[int, 'glass:nbins']
Cosmology = Annotated[Any, 'glass:cosmology']

# cls
Cls = dict[tuple[str, str], ArrayLike]
TheoryCls = Annotated[Cls, 'glass:theory_cls']
SampleCls = Annotated[Cls, 'glass:sample_cls']

# random fields
RandomFields = list['RandomField']
RandomMatterFields = Annotated[RandomFields, 'glass:matter']
RandomConvergenceFields = Annotated[RandomFields, 'glass:convergence']

# fields
Fields = ArrayLike
MatterFields = Annotated[Fields, 'glass:matter']
ConvergenceFields = Annotated[Fields, 'glass:convergence']
ShearFields = Annotated[Fields, 'glass:shear']

# tracers
GalaxyFields = Annotated[Fields, 'glass:galaxies']


def get_annotation(hint):
    '''return the extra information from annotated type hints'''

    orig, args = get_origin(hint), get_args(hint)
    name = None

    if orig == Union and len(args) == 2 and type(None) in args:
        hint = args[0] if isinstance(None, args[1]) else args[1]
        orig, args = get_origin(hint), get_args(hint)

    if orig == Annotated:
        for arg in args[1:]:
            if isinstance(arg, str) and arg.startswith('glass:'):
                name = arg.removeprefix('glass:')

    return name


def annotate(func, name):
    '''annotate a function'''

    @wraps(func)
    def annotated_func(*args, **kwargs):
        return func(*args, **kwargs)

    # make a copy of the annotations dict, or we would change original
    try:
        annotated_func.__annotations__ = annotated_func.__annotations__.copy()
    except AttributeError:
        # no annotations in the first place
        annotated_func.__annotations__ = {}

    # get the original annotation of the function
    t = annotated_func.__annotations__.get('return', NoneType)

    # update the return annotation with the new type hint
    annotated_func.__annotations__['return'] = Annotated[t, f'glass:{name}']

    # return the function with the new annotations
    return annotated_func
