# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''module for simulation objects'''

__all__ = [
    'Ref',
    'Call',
    'Simulation',
]


import logging

from typing import NamedTuple, Callable, get_type_hints
from inspect import signature

from .typing import get_annotation, annotate, RandomFields
from .random import (
    collect_cls,
    compute_gaussian_cls,
    regularize_gaussian_cls,
    generate_random_fields,
    field_from_random_fields,
)


log = logging.getLogger('glass.simulation')


class Ref(NamedTuple):
    name: str

    def __repr__(self):
        return self.name


class Call(NamedTuple):
    func: Callable
    args: list
    kwargs: dict

    @classmethod
    def make(cls, state, func, *args, **kwargs):
        # inspect signature bound to given args and kwargs
        sig = signature(func)
        try:
            ba = sig.bind_partial(*args, **kwargs)
        except TypeError as e:
            raise TypeError(f"{e} for function '{func.__name__}'") from None

        # get type hints with annotations
        hints = get_type_hints(func, include_extras=True)

        # get annotations from type hints
        annotations = {par: get_annotation(hint) for par, hint in hints.items()}

        log.debug('annotations for %s: %d', func.__name__, len(annotations))
        for par, ann in annotations.items():
            log.debug('- %s: %s', par, ann)

        # make sure all func parameters can be obtained
        for par in sig.parameters:
            if par in ba.arguments:
                arg = ba.arguments[par]
                if isinstance(arg, Ref) and arg.name not in state:
                    raise NameError(f"parameter '{par}' of function '{func.__name__}' references unknown name '{arg.name}'")
            elif par in annotations and annotations[par].name in state:
                ba.arguments[par] = Ref(annotations[par].name)
            elif sig.parameters[par].default is not sig.parameters[par].empty:
                pass
            else:
                raise TypeError(f"missing argument '{par}' of function '{func.__name__}'")

        return cls(func, ba.args, ba.kwargs)

    def __call__(self, ns):
        args = []
        kwargs = {}
        for arg in self.args:
            if isinstance(arg, Ref):
                arg = ns[arg.name]
            args.append(arg)
        for par, arg in self.kwargs.items():
            if isinstance(arg, Ref):
                arg = ns[arg.name]
            kwargs[par] = arg
        return self.func(*args, **kwargs)

    def __repr__(self):
        name = self.func.__name__
        args = ', '.join([f'{arg!r}' for arg in self.args] + [f'{par}={arg!r}' for par, arg in self.kwargs.items()])
        return f'{name}({args})'


class Simulation:
    def __init__(self, *, nside=None, zbins=None, allow_missing_cls=False):
        self._cosmology = None
        self._cls = None
        self._random = {}
        self._steps = []

        self.allow_missing_cls = allow_missing_cls

        self.state = {}
        if nside is not None:
            self.state['nside'] = nside
        if zbins is not None:
            self.state['zbins'] = zbins
            self.state['nbins'] = len(zbins) - 1

    def _add_call(self, name, call):
        # if result has a name, make it known to the simulation state
        if name:
            # warn before overwriting
            if name in self.state:
                log.warning('overwriting "%s" with %s', name, call)

            # set state to None initially, any placeholder value would do
            self.state[name] = None

        # return name and function call
        self._steps.append((name, call))

    def _add_random(self, name, call):
        # at the first random field, set up the machinery to sample:
        # - collect the random fields
        # - collect the cls for the random fields
        # - transform the cls to Gaussian cls
        # - regularize the Gaussian cls
        # - sample the random fields from the regularized Gaussian cls
        # - transform regularized Gaussian cls to regularized cls (optional)
        if len(self._random) == 0:
            self.add(self.collect_random_fields)
            self.add(collect_cls, allow_missing=self.allow_missing_cls)
            self.add(compute_gaussian_cls)
            self.add(regularize_gaussian_cls)
            self.add(generate_random_fields)

        # store the added random field
        self._random[name] = call

        # add a call to assign the random field to the given name
        self.add(annotate(field_from_random_fields, name=name), name)

    def add(self, func, *args, **kwargs):
        '''add a function to the simulation'''

        call = Call.make(self.state, func, *args, **kwargs)

        # get the return information of the function:
        # - if there is no '__annotations__' in func return an empty dict
        # - if there is no 'return' in func.__annotations__ return None
        # - get_annotation(None) returns an empty annotation
        ret = get_annotation(getattr(func, '__annotations__', {}).get('return', None))

        # get name of output from annotation
        name = ret.name

        # random fields require extra steps
        if ret.random:
            self._add_random(name, call)
        else:
            self._add_call(name, call)

        return name, call

    @property
    def nside(self):
        '''nside of the simulation'''

        if 'nside' not in self.state:
            raise AttributeError('simulation does not have nside')
        return self.state['nside']

    @property
    def zbins(self):
        '''redshift bins of the simulation'''

        if 'zbins' not in self.state:
            raise AttributeError('simulation does not have zbins')
        return self.state['zbins']

    @property
    def nbins(self):
        '''number of redshift bins in the simulation'''

        if 'nbins' not in self.state:
            raise AttributeError('simulation does not have nbins')
        return self.state['nbins']

    def collect_random_fields(self) -> RandomFields:
        '''collect the random fields in the simulation'''

        random_fields = {}
        for field, call in self._random.items():
            transforms = call(self.state)
            for i, t in enumerate(transforms):
                random_fields[f'{field}[{i}]'] = t

        log.debug('collected %d random fields', len(random_fields))
        for name, transform in random_fields.items():
            log.debug('- %s: %s', name, transform)

        return random_fields

    def run(self):
        '''run the simulation'''

        log.info('# simulate')

        for name, call in self._steps:
            if name:
                log.info('## %s = %s', name, call)
            else:
                log.info('## %s', call)

            # call the computation, resolving references in the state
            # store the result in the state
            self.state[name] = call(self.state)

        # return the state as the result of the run
        return self.state
