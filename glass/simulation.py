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
from functools import cached_property

from .types import get_annotation
from .cls import collect_cls
from .random import generate_random_fields


log = logging.getLogger('glass.simulation')


class Ref(t.NamedTuple):
    name: str

    def __repr__(self):
        return self.name


class Call(t.NamedTuple):
    func: t.Callable
    args: t.Sequence
    kwargs: t.Mapping

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
    def __init__(self, *, nside=None, zbins=None):
        self._cls = None
        self._random = {}
        self._fields = {}

        self.state = {}
        if nside is not None:
            self.state['nside'] = nside
        if zbins is not None:
            self.state['zbins'] = zbins
            self.state['nbins'] = len(zbins) - 1

    def _make_call(self, name, func, args, kwargs):
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

        # resolve default name if None given
        if name is None and 'return' in annotations:
            name = annotations['return'].name
        if name is None:
            raise TypeError(f'cannot infer name of unnamed function "{func.__name__}"')

        # make sure all func parameters can be obtained
        for par in sig.parameters:
            if par in ba.arguments:
                arg = ba.arguments[par]
                if isinstance(arg, Ref) and arg.name not in self.state:
                    raise NameError(f"parameter '{par}' for function '{func.__name__}' of {name} references unknown name '{arg.name}'")
            elif par in annotations and annotations[par].name in self.state:
                ba.arguments[par] = Ref(annotations[par].name)
            elif sig.parameters[par].default is not sig.parameters[par].empty:
                pass
            else:
                raise TypeError(f"missing argument '{par}' for function '{func.__name__}' of {name}")

        return name, Call(func, ba.args, ba.kwargs), annotations.get('return', None)

    def set_cosmology(self, cosmology):
        '''set the cosmology for the simulation'''

        self.state['cosmology'] = cosmology

    def set_cls(self, func, *args, **kwargs):
        '''set the cls for the simulation'''

        name, self._cls, _ = self._make_call(None, func, args, kwargs)

        self.state[name] = None

        return name, self._cls

    def add_field(self, name, func, *args, **kwargs):
        '''add a field to the simulation'''

        name, call, return_info = self._make_call(name, func, args, kwargs)

        if name in self.state:
            log.warning('overwriting "%s" with %s', name, call)

        if getattr(return_info, 'random', False):
            self._random[name] = call
        else:
            self._fields[name] = call

        self.state[name] = None

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

    @property
    def cls(self):
        '''cls for the simulation'''

        if 'cls' not in self.state:
            raise AttributeError('simulation does not have cls')

        if self.state['cls'] is None:
            log.info('obtaining cls...')

            self.state['cls'] = self._cls(self.state)

            log.debug('obtained %d cls:', len(self.state['cls']))
            for a, b in self.state['cls'].keys():
                log.debug('- %s-%s', a, b)

        return self.state['cls']

    @cached_property
    def fields(self):
        '''generate the fields of the simulation'''

        log.info('simulating fields...')

        # this will contain all fields by name
        fields = {}

        # number of random fields
        nrandom = len(self._random)

        log.debug('random fields: %d', nrandom)

        # random fields need to be generated first, and all together
        if nrandom > 0:
            # metadata
            nside = self.nside
            nbins = self.nbins

            # create the RandomField instances which describe the random fields
            # to the generate_random_fields function
            random_names, random_fields = [], []
            for field, call in self._random.items():
                log.debug('- %s:', field)

                rns = [f'{field}[{i}]' for i in range(nbins)]
                rfs = call(self.state)

                if len(rfs) != nbins:
                    raise TypeError(f'random field "{field}" returned {len(rfs)} item(s) for {nbins} bin(s)')

                for rn, rf in zip(rns, rfs):
                    log.debug('  - %s: %s', rn, rf)

                random_names += rns
                random_fields += rfs

            # collect the cls, this also computes the cls if not done before
            cls = collect_cls(random_names, self.cls)

            log.info('generating random fields...')
            for field in self._random:
                log.info('- %s', field)

            # sample maps for all of these random fields
            random_maps = generate_random_fields(nside, random_fields, cls)

            log.debug('shape of random maps: %s', np.shape(random_maps))

            # reshape to number of fields x number of bins
            random_maps = np.reshape(random_maps, (nrandom, nbins, -1))

            log.debug('reshaped random maps: %s', np.shape(random_maps))

            # store the generated maps in the fields for returning
            # also store all random fields in the state for subsequent calls
            for field, m in zip(self._random.keys(), random_maps):
                fields[field] = m
                self.state[field] = m

        log.debug('fields: %d', len(self._fields))

        log.info('generating fields...')

        # now generate all other fields one by one
        for field, call in self._fields.items():
            log.info('- %s', field)

            # call the computation, resolving references in the state
            m = call(self.state)

            # store the map in the fields for returning
            fields[field] = m

            # store the map in the state for subsequent computations
            self.state[field] = m

        # returning the fields caches this property until it is deleted
        return fields
