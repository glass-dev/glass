# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''transformations of Gaussian random maps to other distributions'''

__all__ = [
    'RandomField',
    'NormalField',
    'LognormalField',
]


import numpy as np
import gaussiancl
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, fields as dataclass_fields
from functools import singledispatchmethod, partial


@dataclass
class RandomField(metaclass=ABCMeta):
    '''abstract base class for transformed Gaussian random fields

    Abstract base class for fields that are generated via transformation of
    Gaussian random fields.  Gaussian random fields are generated from Cls, and
    random field transformations must also provide the transformations of the
    intended transformed field Cls to Gaussian Cls.

    The actual transformations of the random maps are always applied in place.

    '''

    mean: float = 0.0

    @classmethod
    def parameters(cls):
        return tuple(f.name for f in dataclass_fields(cls))

    @abstractmethod
    def __call__(self, field, var):
        pass

    @abstractmethod
    def gaussiancl(self, other):
        return NotImplemented

    def __and__(self, other):
        return self.gaussiancl(other)

    def __rand__(self, other):
        return self.gaussiancl(other)


@dataclass
class NormalField(RandomField):
    '''normal random field'''

    def __call__(self, field, var):
        # add mean to Gaussian field
        field += self.mean
        return field

    @singledispatchmethod
    def gaussiancl(self, other):
        if type(other) != type(self):
            return NotImplemented

        return lambda cl, inv=False: cl


@dataclass
class LognormalField(RandomField):
    '''lognormal random field'''

    shift: float = 1.0

    def __call__(self, field, var):
        # fix the mean of the Gaussian field
        field += np.log(self.mean + self.shift) - var/2
        # exponentiate values in place
        np.exp(field, out=field)
        # lognormal shift
        field -= self.shift
        return field

    @singledispatchmethod
    def gaussiancl(self, other):
        if type(other) != type(self):
            return NotImplemented

        alpha = self.mean + self.shift
        alpha2 = other.mean + other.shift
        return partial(gaussiancl.lognormal_cl, alpha=alpha, alpha2=alpha2)

    @gaussiancl.register
    def _(self, other: NormalField):
        alpha = self.mean + self.shift
        return partial(gaussiancl.lognormal_normal_cl, alpha=alpha)
