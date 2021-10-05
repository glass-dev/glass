# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''module for simulation objects'''

__all__ = [
    'Ref',
    'Call',
    'Simulation',
]


import typing as t
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

        # get default refs of func from annotated type hints
        default_refs = {par: get_default_ref(ann) for par, ann in hints.items()}

        log.debug('default refs for %s: %d', func.__name__, len(default_refs))
        for par, ref in default_refs.items():
            log.debug('- %s: %s', par, ref)

        # resolve default name if None given
        if name is None:
            name = default_refs.get('return', None)
        if name is None:
            raise TypeError(f'cannot infer name of unnamed function "{func.__name__}"')

        # make sure all func parameters can be obtained
        for par in sig.parameters:
            if par in ba.arguments:
                arg = ba.arguments[par]
                if isinstance(arg, Ref) and arg.name not in self.state:
                    raise NameError(f"parameter '{par}' for function '{func.__name__}' of {name} references unknown name '{arg.name}'")
            elif par in default_refs and default_refs[par] in self.state:
                ba.arguments[par] = Ref(default_refs[par])
            elif sig.parameters[par].default is not sig.parameters[par].empty:
                pass
            else:
                raise TypeError(f"missing argument '{par}' for function '{func.__name__}' of {name}")

        return name, Call(func, ba.args, ba.kwargs)

    def set_cosmology(self, cosmology):
        '''set the cosmology for the simulation'''

        self.state['cosmology'] = cosmology

    def set_cls(self, func, *args, **kwargs):
        '''set the cls for the simulation'''

        name, self._cls = self._make_call(None, func, args, kwargs)

        self.state[name] = None

        return name, self._cls

    def add_field(self, name, func, *args, **kwargs):
        '''add a field to the simulation'''

        hints = t.get_type_hints(func)

        log.debug('type hints for %s: %d', func.__name__, len(hints))
        for par, hint in hints.items():
            log.debug('- %s: %s', par, hint if hint is not ArrayLike else 'ArrayLike')

        # check if random field
        return_type = hints.get('return', None)
        if return_type is not None:
            is_random = isinstance(return_type, type) and issubclass(return_type, RandomField)
        else:
            is_random = False

        log.debug('%s is random: %s', func.__name__, is_random)

        name, call = self._make_call(name, func, args, kwargs)

        if name in self.state:
            log.warning('overwriting "%s" with %s', name, call)

        if is_random:
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
                log.debug('- (%s, %s)', a, b)

        return self.state['cls']

    @cached_property
    def fields(self):
        '''generate the fields of the simulation'''

        log.info('simulating fields...')

        # this will contain all fields by name
        fields = {}

        log.debug('random fields: %d', len(self._random))

        # random fields need to be generated first, and all together
        if len(self._random) > 0:
            # create the RandomField instances which describe the random fields
            # to the generate_random_fields function
            random = {}
            for field, call in self._random.items():
                random[field] = call(self.state)

                log.debug('- %s: %s', field, random[field])

            # get the metadata, this gets the cls if not done previously
            nside = self.nside
            nbins = self.nbins
            cls = self.cls

            log.info('generating random fields...')
            for field in random:
                log.info('- %s', field)

            random = generate_random_fields(nside, random, nbins, cls)

            # store the generated maps in the fields for returning
            fields.update(random)

            # also store all random fields in the state for subsequent calls
            self.state.update(random)

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
