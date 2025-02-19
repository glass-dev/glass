Random fields
=============

.. currentmodule:: glass

The following functions provide functionality for simulating random fields on
the sphere. This is done in the form of HEALPix maps.


Angular power spectra
---------------------

.. _twopoint_order:

All functions that process sets of two-point functions expect them as a
sequence using the following "Christmas tree" ordering:

.. raw:: html
   :file: figures/spectra_order.svg

In other words, the sequence begins as such:

* Index 0 describes the auto-correlation of field 0,
* index 1 describes the auto-correlation of field 1,
* index 2 describes the cross-correlation of field 1 and field 0,
* index 3 describes the auto-correlation of field 2,
* index 4 describes the cross-correlation of field 2 and field 1,
* index 5 describes the cross-correlation of field 2 and field 0,
* etc.

In particular, two-point functions for the first :math:`n` fields are contained
in the first :math:`T_n = n \, (n + 1) / 2` entries of the sequence.

To easily generate or iterate over sequences of two-point functions in standard
order, see the :func:`glass.enumerate_spectra` and
:func:`glass.spectra_indices` functions.


Preparing inputs
----------------

.. autofunction:: discretized_cls
.. autofunction:: compute_gaussian_spectra
.. autofunction:: solve_gaussian_spectra


Expectations
------------

.. autofunction:: effective_cls


Generating fields
-----------------

.. autofunction:: generate


Lognormal fields
----------------

.. autofunction:: lognormal_fields

GLASS comes with the following functions for setting accurate lognormal shift
values:

.. autofunction:: lognormal_shift_hilbert2011


Regularisation
--------------

When sets of angular power spectra are used to sample random fields, their
matrix :math:`C_\ell^{ij}` for fixed :math:`\ell` must form a valid
positive-definite covariance matrix.  This is not always the case, for example
due to numerical inaccuracies, or transformations of the underlying fields
[Xavier16]_.

Regularisation takes sets of spectra which are ill-posed for sampling, and
returns sets which are well-defined and, in some sense, "close" to the input.

.. autofunction:: regularized_spectra

.. function:: regularized_spectra(..., method="nearest", tol=None, niter=100)
   :no-index:

   Compute the (possibly defective) correlation matrices of the given spectra,
   then find the nearest valid correlation matrices, using the alternating
   projections algorithm of [Higham02]_ with tolerance *tol* for *niter*
   iterations.  This keeps the diagonals (i.e.  auto-correlations) fixed, but
   requires all of them to be nonnegative.

   .. seealso::

      :func:`glass.core.algorithm.cov_nearest`
         Equivalent function for covariance matrices.

      :func:`glass.core.algorithm.nearcorr`
         Nearest correlation matrix.

.. function:: regularized_spectra(..., method="clip", rtol=None)
   :no-index:

   Clip negative eigenvalues of the spectra's covariance matrix to zero.  This
   is a simple fix that guarantees positive semi-definite spectra, but can
   affect the spectra significantly.

   .. seealso::

      :func:`glass.core.algorithm.cov_clip`
         Equivalent function for covariance matrices.


Indexing
--------

.. autofunction:: getcl
.. autofunction:: enumerate_spectra
.. autofunction:: spectra_indices
.. autofunction:: glass_to_healpix_spectra
.. autofunction:: healpix_to_glass_spectra
.. autofunction:: cov_from_spectra


Deprecated
----------

.. autofunction:: lognormal_gls
.. autofunction:: generate_gaussian
.. autofunction:: generate_lognormal
