.. module:: glass.rng

:mod:`glass.rng` --- Random number generation utilities
========================================================

.. currentmodule:: glass.rng

This module includes functions for dispatching random number generators using
consistent seeds. The choice of rng generator is determined based on the array
library chosen by the user.

This module should be imported manually if used outside of GLASS::

    import glass.rng


General
-------

.. autofunction:: default_rng
