# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''
===================================
Matter fields (:mod:`glass.matter`)
===================================

.. currentmodule:: glass.matter

Generators
==========

Random fields
-------------

.. autosummary::
   :template: generator.rst
   :toctree: generated/

   lognormal_matter

'''

from .core import generator
from .random import generate_lognormal, generate_normal


@generator('cl -> delta')
def lognormal_matter(nside, rng=None):
    '''generate lognormal matter fields from Cls'''
    yield from generate_lognormal(nside, shift=1., rng=rng)


@generator('cl -> delta')
def gaussian_matter(nside, rng=None):
    '''generate Gaussian matter fields from Cls'''
    yield from generate_normal(nside, rng=rng)
