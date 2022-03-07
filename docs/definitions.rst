===========
Definitions
===========

.. glossary::

   ellipticity (modulus)
      If :math:`q = b/a` is the axis ratio of an elliptical isophote with
      semi-major axis :math:`a` and semi-minor axis :math:`b`, the ellipticity
      modulus is :math:`\epsilon = (1 - q)/(1 + q)`.  The orientation of the
      ellipticity is described by the :term:`ellipticity (complex)`.

   ellipticity (complex)
      If :math:`q = b/a` is the axis ratio of an elliptical isophote with
      semi-major axis :math:`a` and semi-minor axis :math:`b`, and :math:`\phi`
      is the orientation of the elliptical isophote, the complex-valued
      ellipticity is

      .. math::

         \epsilon = \frac{1 - q}{1 + q} \, \mathrm{e}^{\mathrm{i} \, 2\phi} \;.

      For ellipticity without the orientation, see :term:`ellipticity
      (modulus)`.

   visibility
      The visibility is defined as the *a priory* probability of observing an
      object in a given point of the sky.  As such, the visibility is a number
      between 0 and 1.

   visibility map
      A visibility map is a HEALPix map that describes the *a priori*
      probability of observing an object inside a given HEALPix pixel, with
      pixel values between 0 and 1.  It is hence the averaged, not integrated,
      map of the :term:`visibility`.
