# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''module for simulation objects'''

__all__ = [
    'Simulation',
]


import logging
import os
import time

from typing import NamedTuple, Callable, get_type_hints
from inspect import signature
from collections.abc import Mapping, Sequence

from .typing import get_annotation, NSide, TheoryCls, RandomFields
from .random import generate_random_fields


log = logging.getLogger('glass.simulation')


def _call_str(func, args, kwargs):
    '''pretty print a function call'''
    name = func.__name__
    args = ', '.join([f'{arg!r}' for arg in args] + [f'{par}={arg!r}' for par, arg in kwargs.items()])
    return f'{name}({args})'


def mkworkdir(workdir, run):
    '''create the working directory for a run'''

    rundir = os.path.join(workdir, run)
    lastrun = os.path.join(workdir, 'lastrun')
    os.makedirs(rundir, exist_ok=False)
    with open(lastrun, 'a') as f:
        f.write(f'{run}\n')
    return rundir


class Ref(NamedTuple):
    state: dict
    name: str

    @staticmethod
    def resolve(obj):
        if isinstance(obj, Ref):
            return obj()
        elif isinstance(obj, str):
            return obj
        elif isinstance(obj, Sequence):
            return obj.__class__(Ref.resolve(item) for item in obj)
        elif isinstance(obj, Mapping):
            return obj.__class__((key, Ref.resolve(value)) for key, value in obj.items())
        else:
            return obj

    def __call__(self):
        return self.state[self.name]

    def __repr__(self):
        return self.name


class Call(NamedTuple):
    func: Callable
    args: list
    kwargs: dict
    name: str

    def __call__(self):
        args = []
        kwargs = {}
        for arg in self.args:
            args.append(Ref.resolve(arg))
        for par, arg in self.kwargs.items():
            kwargs[par] = Ref.resolve(arg)
        return self.func(*args, **kwargs)

    def __repr__(self):
        r = f'{self.name} = ' if self.name is not None else ''
        r += _call_str(self.func, self.args, self.kwargs)
        return r


class Simulation:
    def __init__(self, *, workdir=None, nside=None, lmax=None, zbins=None, allow_missing_cls=False):
        self._workdir = workdir
        self._random_fields = []
        self._steps = []

        self.allow_missing_cls = allow_missing_cls

        # initialise the state with some simple properties
        self.state = {}
        if workdir is not None:
            self.state['workdir'] = None
        if nside is not None:
            self.state['nside'] = nside
        if lmax is not None:
            self.state['lmax'] = lmax
        if zbins is not None:
            self.state['zbins'] = zbins
            self.state['nbins'] = len(zbins) - 1

    def _make_call(self, func, /, *args, **kwargs):
        # if func is not callable below, the error is hard to understand
        if not callable(func):
            raise TypeError('func is not callable')

        # inspect signature of the function
        sig = signature(func)

        log.debug('updating parameters...')

        # update parameters with defaults from sim
        params = []
        for par in sig.parameters.values():
            log.debug('- %s', par)

            if par.annotation is not par.empty:
                ann = get_annotation(par.annotation)
                try:
                    default = self.ref(ann)
                except KeyError:
                    pass
                else:
                    par = par.replace(default=default)

                    log.debug('  -> %s', par)

            params.append(par)

        # update signature with updated parameters
        sig = sig.replace(parameters=params)

        # now bind the given args and kwargs to the signature
        try:
            ba = sig.bind(*args, **kwargs)
        except TypeError as e:
            raise TypeError(f'{func.__name__}: {e}') from None

        # apply the (updated) defaults in the bound args
        ba.apply_defaults()

        # get the name of the output from the return annotation
        # use signature only if callable itself has no return annotation
        # this is because callable may have overwritten the annotation
        try:
            name = get_annotation(func.__annotations__['return'])
        except (AttributeError, KeyError):
            name = get_annotation(sig.return_annotation)

        log.debug('returns: %s', name)

        # construct the function call
        return Call(func, ba.args, ba.kwargs, name)

    def _add_name(self, name):
        if name in self.state:
            # warn about overwriting
            log.warning('WARNING: overwriting %s', name)
        else:
            # set state to None initially, any placeholder value would do
            self.state[name] = None

    def _add_call(self, call, position=None):
        # if result has a name, make it known to the simulation state
        if call.name:
            self._add_name(call.name)

        # insert call into steps at given position
        if position is None:
            self._steps.append(call)
        else:
            self._steps.insert(position, call)

    def _add_random_fields(self, call):
        # find the index of the random sampling operation in steps
        try:
            pos = next(i for i, s in enumerate(self._steps) if s.func == self.generate_random_fields)
        except StopIteration:
            pos = None

        # insert the step before the start of the random field generation
        self._add_call(call, pos)

        # add this name to the list of random fields
        if call.name:
            self._random_fields.append(call.name)

        # if random field generation is not in steps, add it
        if pos is None:
            pos = len(self._steps)
            self.add(self.generate_random_fields)
            self._add_name('cls')
            self._add_name('gaussian_cls')
            self._add_name('reg_gaussian_cls')
            self._add_name('regularized_cls')

    def generate_random_fields(self, nside: NSide, cls: TheoryCls):
        '''generate the random fields in the simulation'''

        # collect the random fields, which were called before this
        fields = {name: self.state[name] for name in self._random_fields}

        # generate the random fields
        _fields, _cls = generate_random_fields(nside, cls, fields,
                                               allow_missing_cls=self.allow_missing_cls,
                                               return_cls=True)

        # store the outputs in the state
        self.state.update(_fields)
        self.state.update(_cls)

    def add(self, func, *args, **kwargs):
        '''add a function to the simulation'''

        log.debug('adding call: %s', _call_str(func, args, kwargs))

        call = self._make_call(func, *args, **kwargs)

        # random fields require extra steps
        return_type = get_type_hints(func).get('return', None)
        if return_type == RandomFields:
            self._add_random_fields(call)
        else:
            self._add_call(call)

        return call

    def ref(self, name):
        if name not in self.state:
            raise KeyError(f'name "{name}" is undefined')
        return Ref(self.state, name)

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

    def run(self):
        '''run the simulation'''

        log.info('# simulate')

        # set the run id
        self.state['__run__'] = time.strftime('%y%m%d.%H%M%S')

        log.info('run: %s', self.state['__run__'])

        if self._workdir:
            self.state['workdir'] = mkworkdir(self._workdir, self.state['__run__'])

        for call in self._steps:
            log.info('## %s', call)

            # call the computation, resolving references in the state
            result = call()

            # store the result in the state
            if call.name:
                self.state[call.name] = result

        # return the state as the result of the run
        return self.state
