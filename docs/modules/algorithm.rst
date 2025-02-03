.. module:: glass.core.algorithm

:mod:`glass.core.algorithm` --- General purpose algorithms
==========================================================

.. currentmodule:: glass.core.algorithm

This module contains general implementations of algorithms which are used by
GLASS, but are otherwise unrelated to GLASS functionality.

This module must be imported manually::

    import glass.core.algorithm as algo


Non-negative least squares
--------------------------

.. autofunction:: nnls


Nearest correlation matrix
--------------------------

.. autofunction:: nearcorr


Covariance matrix regularisation
--------------------------------

.. autofunction:: cov_clip
.. autofunction:: cov_nearest
