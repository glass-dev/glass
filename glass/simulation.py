# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''module for simulation objects'''

__all__ = [
    'Ref',
    'Call',
    'Simulation',
]


import numpy as np
import typing as t
import logging

from inspect import signature

from .typing import get_annotation, NSide, ClsDict
from .cls import collect_cls
from .random import compute_gaussian_cls, regularize_gaussian_cls, generate_random_fields


log = logging.getLogger('glass.simulation')


class Ref(t.NamedTuple):
    name: str

    def __repr__(self):
        return self.name


class Call(t.NamedTuple):
    func: t.Callable
    args: t.Sequence
    kwargs: t.Mapping

    @classmethod
    def make(cls, state, func, *args, **kwargs):
        # inspect signature bound to given args and kwargs
        sig = signature(func)
        try:
            ba = sig.bind_partial(*args, **kwargs)
        except TypeError as e:
            raise TypeError(f"{e} for function '{func.__name__}'") from None

        # get type hints with annotations
        hints = t.get_type_hints(func, include_extras=True)

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
        # return name and functionc call
        self._steps.append((name, call))

    def _add_random(self, name, call):
        if len(self._random) == 0:
            self._add_call('random', Call.make(self.state, self.simulate_random_fields))
        self._add_call(name, Call.make(self.state, self.random_field, name))
        self._random[name] = call

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

        # if result has a name, make it known to the simulation state
        if name:
            # warn before overwriting
            if name in self.state:
                log.warning('overwriting "%s" with %s', name, call)

            # set state to None initially, any placeholder value would do
            self.state[name] = None

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

    def simulate_random_fields(self, nside: NSide, cls: ClsDict):
        '''simulate the random fields in the simulation'''

        # create the RandomField instances which describe the random fields
        # to the generate_random_fields function
        offsets, fields, names = [], [], []
        for field, call in self._random.items():
            offsets.append(len(fields))
            fields += call(self.state)
            names += [f'{field}[{i}]' for i in range(len(fields)-len(names))]
        offsets.append(len(names))

        log.debug('collecting cls...')

        cls = collect_cls(names, cls, allow_missing=self.allow_missing_cls)

        log.debug('collected %d cls, of which %d are None', len(cls), sum(cl is None for cl in cls))

        log.debug('computing Gaussian cls...')

        gaussian_cls = compute_gaussian_cls(cls, fields, nside)

        log.debug('regularising Gaussian cls...')

        regularized_cls = regularize_gaussian_cls(gaussian_cls)

        log.info('generating random fields...')
        for i, (name, field) in enumerate(zip(names, fields)):
            log.info('- %d: %s = %s', i, name, field)

        maps = generate_random_fields(nside, regularized_cls, fields)

        log.debug('shape of random maps: %s', np.shape(maps))
        log.debug('assigning to fields...')

        # assign maps to fields
        random = {}
        for name, first, last in zip(self._random, offsets, offsets[1:]):
            random[name] = maps[first:last]

            log.debug('- %s: %d', name, last-first)

        # return the dict of fields and their random maps
        return random

    def random_field(self, name):
        '''return a random field in the simulation'''
        return self.state['random'].get(name)

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
