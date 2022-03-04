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

from ._generator import generator
from .random import lognormal_random_fields


@generator('cl -> delta')
def lognormal_matter(nside, rng=None):
    '''generate lognormal matter fields from Cls'''
    return lognormal_random_fields(nside, shift=1., rng=rng)
