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


Indexing
--------

.. autofunction:: getcl
.. autofunction:: enumerate_spectra
.. autofunction:: spectra_indices


Deprecated
----------

.. autofunction:: lognormal_gls
.. autofunction:: generate_gaussian
.. autofunction:: generate_lognormal
