# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''module for generator definition'''

from collections.abc import Generator
from functools import wraps


class WrappedGenerator(Generator):
    '''wrapper for generators'''

    __slots__ = 'generator', 'receives', 'yields'

    def __init__(self, generator, receives=None, yields=None):
        '''wrap a generator'''
        self.generator = generator
        self.receives = receives
        self.yields = yields

    def __iter__(self):
        '''call iter() on wrapped generator'''
        return iter(self.generator)

    def __next__(self):
        '''call next() on wrapped generator'''
        return next(self.generator)

    def send(self, value):
        '''call send() on wrapped generator'''
        return self.generator.send(value)

    def throw(self, value):
        '''call throw() on wrapped generator'''
        return self.generator.throw(value)

    def close(self):
        '''call close() on wrapped generator'''
        return self.generator.close()

    def __getattr__(self, name):
        '''get an attribute of the wrapped generator'''
        if name in self.__slots__:
            obj = self
        else:
            obj = self.generator
        return object.__getattribute__(obj, name)

    def __setattr__(self, name, value):
        '''set an attribute of the wrapped generator'''
        if name in self.__slots__:
            obj = self
        else:
            obj = self.generator
        object.__setattr__(obj, name, value)


def generator(receives=None, yields=None):
    '''decorator to wrap a generator function'''
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            g = f(*args, **kwargs)
            g.__name__ = wrapper.__name__
            return WrappedGenerator(g, receives, yields)
        return wrapper
    return decorator


def optional(name):
    '''mark generator input as optional'''
    return f'{name}?'
