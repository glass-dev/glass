================================
Galaxies (:mod:`glass.galaxies`)
================================

.. currentmodule:: glass.galaxies


Galaxy bias
===========

Variables
---------

.. autodata:: B
.. autodata:: BFN


Generators
----------

.. autosummary::
   :template: generator.rst
   :toctree: generated/
   :nosignatures:

   gal_b_const
   gal_b_eff
   gal_bias_linear
   gal_bias_loglinear
   gal_bias_function


Galaxy distribution
===================

Variables
---------

.. autodata:: NGAL
.. autodata:: NZ
.. autodata:: GAL_LEN
.. autodata:: GAL_LON
.. autodata:: GAL_LAT


Generators
----------

.. autosummary::
   :template: generator.rst
   :toctree: generated/
   :nosignatures:

   gal_density_const
   gal_density_dndz
   gal_positions_mat
   gal_positions_unif


Galaxy redshifts
================

Variables
---------

.. autodata:: GAL_Z
.. autodata:: GAL_POP


Generators
----------

.. autosummary::
   :template: generator.rst
   :toctree: generated/
   :nosignatures:

   gal_redshifts_nz


Galaxy ellipticities
====================

Variables
---------

.. autodata:: GAL_ELL


Generators
----------

.. autosummary::
   :template: generator.rst
   :toctree: generated/
   :nosignatures:

   gal_ellip_gaussian
   gal_ellip_intnorm
   gal_ellip_ryden04


Other
-----

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ellipticity_ryden04


Galaxy shears
=============

Variables
---------

.. autodata:: GAL_SHE


Generators
----------

.. autosummary::
   :template: generator.rst
   :toctree: generated/
   :nosignatures:

   gal_shear_interp


Photometric redshifts
=====================

Variables
---------

.. autodata:: GAL_PHZ


Generators
----------

.. autosummary::
   :template: generator.rst
   :toctree: generated/
   :nosignatures:

   gal_phz_gausserr
