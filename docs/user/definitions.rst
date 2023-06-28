===========
Definitions
===========

The *GLASS* code uses the following mathematical definitions.

.. glossary::

   deflection
      The deflection :math:`\alpha` is a complex value with spin weight
      :math:`1`.  It describes the displacement of a position along a geodesic
      (i.e. great circle).  The angular distance of the displacement is the
      absolute value :math:`|\alpha|`.  The direction of the displacement is
      the angle given by the complex argument :math:`\arg\alpha`, such that
      :math:`\arg\alpha = 0^\circ` is north, :math:`\arg\alpha = 90^\circ` is
      east, :math:`\arg\alpha = 180^\circ` is south, and :math:`\arg\alpha =
      -90^\circ` is west.

   ellipticity
      If :math:`q = b/a` is the axis ratio of an elliptical isophote with
      semi-major axis :math:`a` and semi-minor axis :math:`b`, and :math:`\phi`
      is the orientation of the elliptical isophote, the complex-valued
      ellipticity is

      .. math::

         \epsilon = \frac{1 - q}{1 + q} \, \mathrm{e}^{\mathrm{i} \, 2\phi} \;.

   pixel window function
      The convolution kernel that describes the shape of pixels in a spherical
      map.  No discretisation of the sphere has pixels of exactly the same
      shape, and the pixel window function is therefore an approximation: It is
      an effective kernel :math:`w_l` such that the discretised map :math:`F`
      of a spherical function :math:`f` has spherical harmonic expansion

      .. math::

         F_{lm} \approx w_l \, f_{lm} \;.

   radial window
      A radial window consist of a window function that assigns a weight
      :math:`W(z)` to each redshift :math:`z` along the line of sight.  Each
      radial window has an associated effective redshift :math:`z_{\rm eff}`
      which could be e.g. the central or mean redshift of the weight function.

      A set of window functions :math:`W_1, W_2, \ldots`, defines the shells of
      the simulation.

   spherical function
      A spherical function :math:`f` is a function that is defined on the
      sphere.  Function values are usually parametrised in terms of spherical
      coordinates, :math:`f(\theta, \phi)`, or using a unit vector,
      :math:`f(\hat{n})`.

   spherical harmonic expansion
      A scalar :term:`spherical function` :math:`f` can be expanded into the
      spherical harmonics :math:`Y_{lm}`,

      .. math::

         f(\hat{n}) = \sum_{lm} f_{lm} \, Y_{lm}(\hat{n}) \;.

      If :math:`f` is not scalar but has non-zero spin weight :math:`s`, it can
      be expanded into the spin-weighted spherical harmonics :math:`{}_sY_{lm}`
      instead,

      .. math::

         f(\hat{n}) = \sum_{lm} f_{lm} \, {}_sY_{lm}(\hat{n}) \;.

   visibility map
      A visibility map describes the *a priori* probability of observing an
      object inside a given pixel, with pixel values between 0 and 1.
