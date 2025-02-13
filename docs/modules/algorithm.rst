.. module:: glass.algorithm

:mod:`glass.algorithm` --- General purpose algorithms
=====================================================

.. currentmodule:: glass.algorithm

This module contains general implementations of algorithms which are used by
GLASS, but are otherwise unrelated to GLASS functionality.

This module should be imported manually if used outside of GLASS::

    import glass.algorithm


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
