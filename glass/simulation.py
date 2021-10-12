# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''module for simulation objects'''

__all__ = [
    'Ref',
    'Call',
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
    name: str

    @staticmethod
    def check(obj, ns):
        if isinstance(obj, Ref):
            if obj.name not in ns:
                raise NameError(f"name '{obj.name}' is not defined")
            return True
        elif isinstance(obj, str):
            return True
        elif isinstance(obj, Sequence):
            return all(Ref.check(item, ns) for item in obj)
        elif isinstance(obj, Mapping):
            return all(Ref.check(value, ns) for value in obj.values())
        else:
            return True

    @staticmethod
    def resolve(obj, ns):
        if isinstance(obj, Ref):
            if obj.name not in ns:
                raise NameError(f"name '{obj.name}' is not defined")
            return ns[obj.name]
        elif isinstance(obj, str):
            return obj
        elif isinstance(obj, Sequence):
            return obj.__class__(Ref.resolve(item, ns) for item in obj)
        elif isinstance(obj, Mapping):
            return obj.__class__((key, Ref.resolve(value, ns)) for key, value in obj.items())
        else:
            return obj

    def __repr__(self):
        return self.name


class Call(NamedTuple):
    func: Callable
    args: list
    kwargs: dict
    name: str

    @classmethod
    def make(cls, /, state, func, *args, **kwargs):
        # if func is not callable below, the error is hard to understand
        if not callable(func):
            raise TypeError('func is not callable')

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

        # the return annotation
        returns = annotations.pop('return', None)

        # make sure all func parameters can be obtained
        for par in sig.parameters:
            if par in ba.arguments:
                arg = ba.arguments[par]
                try:
                    Ref.check(arg, state)
                except NameError as e:
                    raise NameError(f"parameter '{par}' of function '{func.__name__}': {e}") from None
            elif par in annotations:
                arg = Ref(annotations[par])
                try:
                    Ref.check(arg, state)
                except NameError:
                    pass
                else:
                    ba.arguments[par] = arg
            if par not in ba.arguments and sig.parameters[par].default is sig.parameters[par].empty:
                raise TypeError(f"missing argument '{par}' of function '{func.__name__}'")

        return cls(func, ba.args, ba.kwargs, returns)

    def __call__(self, ns):
        args = []
        kwargs = {}
        for arg in self.args:
            args.append(Ref.resolve(arg, ns))
        for par, arg in self.kwargs.items():
            kwargs[par] = Ref.resolve(arg, ns)
        return self.func(*args, **kwargs)

    def __repr__(self):
        r = f'{self.name} = ' if self.name is not None else ''
        r += _call_str(self.func, self.args, self.kwargs)
        return r


class Simulation:
    def __init__(self, *, workdir=None, nside=None, zbins=None, allow_missing_cls=False):
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
        if zbins is not None:
            self.state['zbins'] = zbins
            self.state['nbins'] = len(zbins) - 1

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

        call = Call.make(self.state, func, *args, **kwargs)

        # random fields require extra steps
        return_type = get_type_hints(func).get('return', None)
        if return_type == RandomFields:
            self._add_random_fields(call)
        else:
            self._add_call(call)

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
            result = call(self.state)

            # store the result in the state
            if call.name:
                self.state[call.name] = result

        # return the state as the result of the run
        return self.state
