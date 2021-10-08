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

from .types import get_annotation
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
        self._steps = {}

        self.allow_missing_cls = allow_missing_cls

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

        # annotation for return value information
        return_info = annotations.get('return', get_annotation(None))

        # resolve default name if None given
        if name is None:
            name = return_info.name
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

        return name, Call(func, ba.args, ba.kwargs), return_info

    def add(self, name, func, *args, **kwargs):
        '''add a function to the simulation'''

        name, call, return_info = self._make_call(name, func, args, kwargs)

        if name in self.state:
            log.warning('overwriting "%s" with %s', name, call)

        if name == 'cosmology':
            self._cosmology = call
        elif name == 'cls':
            self._cls = call
        elif return_info.random:
            self._random[name] = call
        else:
            self._steps[name] = call

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

    def run(self):
        '''run the simulation'''

        log.info('simulating...')

        # construct cosmology if given
        if self._cosmology is not None:
            log.info('cosmology...')
            self.state['cosmology'] = self._cosmology(self.state)

        # number of random fields
        nrandom = len(self._random)

        log.debug('number of random fields: %d', nrandom)

        # random fields need to be generated first, and all together
        if nrandom > 0:
            # metadata, this computes the cls if not done before
            nside = self.nside
            nbins = self.nbins
            cls = self.cls

            # create the RandomField instances which describe the random fields
            # to the generate_random_fields function
            random_names, random_fields = [], []
            for field, call in self._random.items():
                rns = [f'{field}[{i}]' for i in range(nbins)]
                rfs = call(self.state)

                if len(rfs) != nbins:
                    raise TypeError(f'random field "{field}" returned {len(rfs)} item(s) for {nbins} bin(s)')

                random_names += rns
                random_fields += rfs

            log.info('list of random fields:')
            for i, (rn, rf) in enumerate(zip(random_names, random_fields)):
                log.info('- %d: %s = %s', i, rn, rf)

            log.debug('collecting cls...')

            cls = collect_cls(random_names, cls, allow_missing=self.allow_missing_cls)

            log.debug('collected %d cls, of which %d are None', len(cls), sum(cl is None for cl in cls))

            log.info('computing Gaussian cls...')

            gaussian_cls = compute_gaussian_cls(cls, random_fields, nside)

            log.info('regularising Gaussian cls...')

            regularized_cls = regularize_gaussian_cls(gaussian_cls)

            log.info('generating random fields...')
            for field in self._random:
                log.info('- %s', field)

            random_maps = generate_random_fields(nside, regularized_cls, random_fields)

            log.debug('shape of random maps: %s', np.shape(random_maps))

            # reshape to number of fields x number of bins
            random_maps = np.reshape(random_maps, (nrandom, nbins, -1))

            log.debug('reshaped random maps: %s', np.shape(random_maps))

            # store all random fields in the state
            for field, m in zip(self._random.keys(), random_maps):
                self.state[field] = m

        log.debug('number of steps: %d', len(self._steps))

        log.info('stepping...')

        # now go through steps one by one
        for name, call in self._steps.items():
            log.info('- %s', name)

            # call the computation, resolving references in the state
            # store the result in the state
            self.state[name] = call(self.state)

        # return the state as the result of the run
        return self.state
