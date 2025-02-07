.. module:: glass.grf

:mod:`glass.grf` --- Gaussian random fields
===========================================

.. currentmodule:: glass.grf


Gaussian angular power spectra
------------------------------

.. autofunction:: compute
.. autofunction:: solve


Transforming correlations
-------------------------

These functions can convert between Gaussian and transformed angular
correlation functions, and form the basis of :func:`glass.grf.compute` and
:func:`glass.grf.solve`.

.. autofunction:: corr
.. autofunction:: icorr
.. autofunction:: dcorr


Transformations
---------------

.. autoprotocol:: Transformation

.. autoclass:: Normal
.. autoclass:: Lognormal
.. autoclass:: SquaredNormal
