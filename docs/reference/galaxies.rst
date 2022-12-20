================================
Galaxies (:mod:`glass.galaxies`)
================================

.. currentmodule:: glass.galaxies


Galaxy bias
===========

Functions
---------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   effective_bias
   linear_bias
   loglinear_bias


Galaxy distribution
===================

Functions
---------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   constant_densities
   densities_from_dndz
   

Generators
----------

.. autodata:: GAL_LEN
.. autodata:: GAL_LON
.. autodata:: GAL_LAT

.. autosummary::
   :template: generator.rst
   :toctree: generated/
   :nosignatures:

   gen_positions_from_matter
   gen_uniform_positions


Galaxy redshifts
================

Generators
----------

.. autodata:: GAL_Z
.. autodata:: GAL_POP

.. autosummary::
   :template: generator.rst
   :toctree: generated/
   :nosignatures:

   gen_redshifts_from_nz
   gen_uniform_redshifts


Galaxy ellipticities
====================

Functions
---------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ellipticity_ryden04


Generators
----------

.. autodata:: GAL_ELL

.. autosummary::
   :template: generator.rst
   :toctree: generated/
   :nosignatures:

   gen_ellip_zero
   gen_ellip_gaussian
   gen_ellip_intnorm
   gen_ellip_ryden04


Galaxy shears
=============

Generators
----------

.. autodata:: GAL_SHE

.. autosummary::
   :template: generator.rst
   :toctree: generated/
   :nosignatures:

   gen_shear_simple
   gen_shear_interp


Photometric redshifts
=====================

Generators
----------

.. autodata:: GAL_PHZ

.. autosummary::
   :template: generator.rst
   :toctree: generated/
   :nosignatures:

   gen_phz_gausserr
