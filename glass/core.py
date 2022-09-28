# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''module for core functionality'''

from functools import wraps


def _parse_signature(s):
    '''parse generator signature into inputs and outputs'''
    i, o, *_ = [list(map(str.strip, _.split(','))) for _ in s.split('->')] + [[]]
    if len(i) == 1:
        i = i[0]
    else:
        while i and i[-1] == '':
            i.pop()
        i = tuple(i)
    if len(o) == 1:
        o = o[0]
    else:
        while o and o[-1] == '':
            o.pop()
        o = tuple(o)
    return i or None, o or None


def _create_signature(inputs, outputs):
    '''create a signature string from inputs and outputs'''
    if inputs is None:
        i = ''
    elif isinstance(inputs, str):
        i = inputs
    else:
        i = ', '.join(map(str, inputs))
    if outputs is None:
        o = ''
    elif isinstance(outputs, str):
        o = '-> ' + outputs
    else:
        o = '-> ' + ', '.join(map(str, outputs))
    if i and o:
        return i + ' ' + o
    else:
        return i + o


def _update_signature(io, names):
    '''update inputs or outputs'''
    if isinstance(io, str):
        return names.get(io, io)
    else:
        return type(io)(names.get(_, _) for _ in io)


class Generator:
    '''wrapper for low-level Python generators'''
    def __init__(self, generator, signature=None, name=None, module=None):
        '''wrap a low-level generator with optional signature'''

        self._generator = generator
        self._name = name
        self._module = module
        self._tags = []
        self._inputs = None
        self._outputs = None

        if signature is not None:
            self._inputs, self._outputs = _parse_signature(signature)

    def __repr__(self):
        return f'Generator({self.qualname}, {self.signature!r})'

    def __str__(self):
        return f'{self.name}: {self.signature}'

    def __iter__(self):
        return iter(self._generator)

    def __getattr__(self, attr):
        return getattr(self._generator, attr)

    @property
    def name(self):
        '''name of the generator'''
        return self._name or '<anonymous>'

    @property
    def module(self):
        '''module where the generator is defined'''
        return self._module or '<anonymous>'

    @property
    def qualname(self):
        '''qualified name of the generator'''
        return f'{self.module}.{self.name}'

    @property
    def tags(self):
        '''tags of the generator'''
        return self._tags

    def tag(self, *tags):
        '''add tags to generator'''
        self._tags += tags

    @property
    def label(self):
        '''label of the generator, including name and tags'''
        return ' - '.join(self.tags + [self.name])

    @property
    def signature(self):
        '''signature of the generator'''
        return _create_signature(self._inputs, self._outputs)

    def inputs(self, **names):
        '''update inputs of generator'''
        self._inputs = _update_signature(self._inputs, names)

    def outputs(self, **names):
        '''update outputs of generator'''
        self._outputs = _update_signature(self._outputs, names)


def generator(signature, *, self=False):
    '''decorator to wrap a low-level generator'''
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            name = getattr(f, '__name__', None)
            module = getattr(f, '__module__', None)
            g = object.__new__(Generator)
            if self:
                args = [g, *args]
            generator = f(*args, **kwargs)
            g.__init__(generator, signature, name, module)
            return g
        return wrapper
    return decorator
