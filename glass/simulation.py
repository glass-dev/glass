# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''module for simulation objects'''

__all__ = [
    'Ref',
    'Call',
    'Simulation',
]


import typing as t
import dataclasses as dc
import logging

from inspect import signature
from functools import cached_property

from .types import ArrayLike, get_default_ref
from .random import RandomField, generate_random_fields


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
    def _make_call(self, origin, func, args, kwargs):
        # inspect signature bound to given args and kwargs
        sig = signature(func)
        try:
            ba = sig.bind_partial(*args, **kwargs)
        except TypeError as e:
            raise TypeError(f"{e} for function '{func.__name__}' of {origin}") from None

        # get type hints with annotations
        hints = t.get_type_hints(func, include_extras=True)
        for par, hint in hints.items():
            # resolve InitVar type from dataclass
            if isinstance(hint, dc.InitVar):
                hints[par] = hint.type

        log.debug('type hints for function %s: %d', func.__name__, len(hints))
        for par, hint in hints.items():
            log.debug('- %s: %s', par, hint if hint is not ArrayLike else 'ArrayLike')

        # get default refs of func from annotated type hints
        default_refs = {par: get_default_ref(ann) for par, ann in hints.items()}

        log.debug('default refs for function %s: %d', func.__name__, len(default_refs))
        for par, ref in default_refs.items():
            log.debug('- %s: %s', par, ref)

        # make sure all func parameters can be obtained
        for par in sig.parameters:
            if par in ba.arguments:
                arg = ba.arguments[par]
                if isinstance(arg, Ref) and arg.name not in self.state:
                    raise NameError(f"parameter '{par}' for function '{func.__name__}' of {origin} references unknown name '{arg.name}'")
            elif par in default_refs and default_refs[par] in self.state:
                ba.arguments[par] = Ref(default_refs[par])
            elif sig.parameters[par].default is not sig.parameters[par].empty:
                pass
            else:
                raise TypeError(f"missing argument '{par}' for function '{func.__name__}' of {origin}")

        return Call(func, ba.args, ba.kwargs)

    def __init__(self, *, nside=None, zbins=None):

        self._cls = None
        self._fields = {}

        self.state = {}
        if nside is not None:
            self.state['nside'] = nside
        if zbins is not None:
            self.state['zbins'] = zbins
            self.state['nbins'] = len(zbins) - 1

    def set_cosmology(self, cosmology):
        '''set the cosmology for the simulation'''

        self.state['cosmology'] = cosmology

    def set_cls(self, func, *args, **kwargs):
        '''set the cls for the simulation'''

        self._cls = self._make_call('cls', func, args, kwargs)
        self.state['cls'] = None
        return self._cls

    def add_field(self, name, func, *args, **kwargs):
        '''add a field to the simulation'''

        call = self._make_call(f"field '{name}'", func, args, kwargs)
        if name in self.state:
            log.warning('overwriting "%s" with %s', name, call)
        self._fields[name] = call
        self.state[name] = None
        return call

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
                log.debug('- (%s, %s)', a, b)

        return self.state['cls']

    @cached_property
    def fields(self):
        '''generate the fields of the simulation'''

        log.info('simulating fields...')

        # this will contain all fields by name
        fields = {}

        # random fields need to be generated first, and all together
        random_fields, nonrandom_fields = {}, {}
        for field, call in self._fields.items():
            # random fields are not functions but subclasses of RandomField
            if isinstance(call.func, type) and issubclass(call.func, RandomField):
                # calling a RandomField produces an instance which describes the
                # random field to the generate_random_fields function
                random_fields[field] = call(self.state)
            else:
                # keep the nonrandom field call for later
                nonrandom_fields[field] = call

        log.debug('random fields: %d', len(random_fields))
        for field, random_field in random_fields.items():
            log.debug('- %s: %s', field, random_field)

        # if there are any random fields at all, generate them
        if len(random_fields) > 0:
            # get the metadata, this gets the cls if not done previously
            nside = self.nside
            nbins = self.nbins
            cls = self.cls

            log.info('generating random fields...')

            randoms = generate_random_fields(nside, random_fields, nbins, cls)

            # store the generated maps in the fields for returning
            fields.update(randoms)

            # also store all random fields in the state for subsequent calls
            self.state.update(randoms)

        log.debug('fields: %d', len(nonrandom_fields))

        # now generate all other fields one by one
        for field, call in nonrandom_fields.items():
            log.debug('- %s: %s', field, call)

            # call the computation, resolving references in the state
            m = call(self.state)

            # store the map in the fields for returning
            fields[field] = m

            # store the map in the state for subsequent computations
            self.state[field] = m

        # returning the fields caches this property until it is deleted
        return fields
