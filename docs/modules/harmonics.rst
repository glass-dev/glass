.. module:: glass.harmonics

:mod:`glass.harmonics` --- Spherical harmonics utilities
========================================================

.. currentmodule:: glass.harmonics

This module contains utilities for working with spherical harmonics which are
used by GLASS, but are otherwise unrelated to GLASS functionality.

This module should be imported manually if used outside of GLASS::

    import glass.harmonics


General
-------

.. autofunction:: multalm


Spherical harmonic transforms
-----------------------------

.. autofunction:: transform
.. autofunction:: inverse_transform
