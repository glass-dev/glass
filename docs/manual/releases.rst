
Release notes
=============

These notes document the changes between individual *GLASS* releases.


2024.2 (15 Nov 2024)
--------------------

* All GLASS user functionality is now available directly from the main
  ``glass`` namespace.  No more imports!

* Changes to the functions dealing with Gaussian spectra:

  * ``gaussian_gls()`` was renamed to :func:`glass.discretized_cls`, which
    better reflects what the function does.

  * The implicit call to :func:`glass.discretized_cls` was removed from
    :func:`glass.lognormal_gls`.  Call :func:`glass.discretized_cls` explicitly
    with the desired ``ncorr=``, ``lmax=``, and ``nside=`` parameters.

* ``redshifts_from_nz()`` is deprecated in favour of :func:`glass.redshifts`,
  as the former is almost never the right choice: the two-point statistics in a
  linear bias model are coming from the shells, so the redshift distribution is
  implicitly the same as the radial profile of the shells.

* Several fixes to :func:`glass.ellipticity_ryden04`.

* Added a FITS catalogue writer tool :func:`glass.write_catalog`.

* Much improved documentation with docstring for all functions.

* Examples now live in the main GLASS repository and documentation.

* GLASS is now fully typed and passes mypy.

* Python 3.13 is now supported.  Python 3.8 and NumPy 1.21 are no longer
  supported.

* The ``glass.core.constants`` module was removed.


2024.1 (16 Jul 2024)
--------------------

* Further changes to the :func:`~glass.shells.partition()` function.

  * The output of ``partition()`` now has the shells axis as its first.  **This
    means that the new output is the transpose of the previous output.**

  * A new ``partition(..., method="nnls")`` method that computes a partition
    with non-negative contributions for each shell.  **This is now the
    default.** The ``"nnls"`` method works much better than ``"lstsq"`` since
    it does not introduce small negative densities, and should almost always be
    preferred.

  * Both ``partition(method="nnls")`` and ``partition(method="lstsq")`` now
    have an additional integral constraint so that the sum of the partition
    recovers the integral of the input function.

  * The ``partition()`` function now works correctly with functions having
    extra axes.

* A new function :func:`~glass.shells.combine()` that evaluates the linear
  combination of radial window functions with given weights.  This function is
  the inverse of :func:`~glass.shells.partition()` and can be used to obtain
  the effect of the discretisation on, e.g., a redshift distribution.

* There is now a way to compute the effective angular power spectra that can
  be expected from a *GLASS* simulation, including all discretisations and
  approximations.

  * A new function :func:`~glass.fields.effective_cls()` which combines power
    spectra using a list of weights.  This function essentially models the
    linear combinations that happen in the simulation.

  * A new function :func:`~glass.points.position_weights()` that returns weights
    for ``effective_cls()`` to model the result of
    :func:`~glass.points.positions_from_delta()`.

  * A new function :func:`~glass.lensing.multi_plane_weights()` that returns
    weights for ``effective_cls()`` to model the result of
    :class:`~glass.lensing.MultiPlaneConvergence`.

* A new function :func:`~glass.galaxies.redshifts()` to sample redshifts
  following a radial window function.  This should always be preferred to the
  existing :func:`~glass.galaxies.redshifts_from_nz()` function, since the
  redshift distribution entering the two-point statistics is in fact fixed by
  the window functions.


2023.7 (1 Aug 2023)
-------------------

* New radial window functions :func:`~glass.shells.linear_windows()` and
  :func:`~glass.shells.cubic_windows()`, which correspond to linear and cubic
  spline interpolation of radial functions, respectively.  These are
  overlapping window functions, and it has been difficult to obtain accurate
  matter power spectra so far.

* The :func:`~glass.shells.partition()` function now returns an array of
  weights to approximate a given function by the window functions.  This is
  necessary to obtain an accurate fit of redshift distributions by overlapping
  window functions.  For example, to get the array of galaxy densities in each
  shells from ``dndz``, one would now do::

      ngal = partition(z, dndz, shells)

* A new function :func:`~glass.fields.getcl()` was added to return angular
  power spectra by index from a list using GLASS ordering.

* The :func:`~glass.galaxies.gaussian_phz()` function now accepts bounds using
  ``lower=`` and ``upper=`` keyword parameters.


2023.6 (30 Jun 2023)
--------------------

- There is some support for simulating the deflections due to weak
  gravitational lensing:

  - The :func:`~glass.lensing.deflect` function applies deflections to
    positions.

  - The :func:`~glass.lensing.from_convergence` function returns one or more
    other lensing fields given the convergence.

  - The ``shear_from_convergence()`` function is deprecated in favour of
    ``from_convergence()``.

- The ``glass`` module is no longer a namespace package.  The new ``glass.ext``
  namespace is reserved for extensions instead.  This is done to follow best
  practices, so that a bad extension can no longer break all of *GLASS* by
  mistake.  The ``glass.all`` meta-module is no longer necessary.

- The point sampling functions :func:`~glass.points.positions_from_delta` and
  :func:`~glass.points.uniform_positions` now return an iterator over points.
  This has lead to orders-of-magnitude improvements in memory use and
  performance when simulating galaxies at Euclid/LSST densities.

- The ellipticity sampling functions :func:`~glass.shapes.ellipticity_gaussian`
  and :func:`~glass.shapes.ellipticity_intnorm` accept array inputs.

- A bug causing incorrect results from :func:`~glass.points.effective_bias` has
  been fixed.


2023.5 (31 May 2023)
--------------------

- The point sampling functions in :mod:`glass.points` now accept extra
  dimensions, and will broadcast leading axes across their inputs.  They also
  return an additional scalar or array with the counts of sampled galaxies.

- The redshift sampling function :func:`glass.galaxies.redshifts_from_nz` now
  supports array input for the ``counts`` argument.  It accepts e.g. the number
  of galaxies returned by the position sampling.

- The profiling functionality in :mod:`glass.user` was removed in favour of
  external packages.


2023.2 (1 Mar 2023)
-------------------

- New user functions :func:`glass.user.save_cls` and
  :func:`glass.user.load_cls` to save and load angular power spectra in the
  *GLASS* format.

- Some type hints were added to library functions.  These are mostly
  perfunctory at this time, but there is interest in adding proper typing
  support in the future, including use of the Array API.

- The ``glass.matter`` module was removed in favour of the more
  appropriately-named :mod:`glass.shells` module for shell definitions.

- Instead of using an array of shell boundaries and separate ``MatterWeights``,
  shells are now entirely defined by a :class:`glass.shells.RadialWindow`
  window function.

- Many functions have an improved interface thanks to the previous point:

  - The ``glass.math.restrict_interval`` function has been replaced by
    :func:`glass.shells.restrict`, as shells are now defined by
    window functions instead of sharp intervals.

  - The :func:`glass.points.effective_bias` function now takes a window
    function as input and computes its effective bias parameter.

  - The ``glass.galaxies.constant_densities`` and ``density_from_dndz``
    functions have been removed, since densities can now easily be partitioned
    by window functions using :func:`glass.shells.restrict` and
    :func:`glass.shells.partition`.

  - The ``zmin`` and ``zmax`` parameters of ``glass.galaxies.redshifts_from_nz``
    have been removed for the same reason.

  - The ``glass.lensing.multi_plane_weights`` function, which computed all
    lensing weights at once, is replaced by the ``add_window`` method of
    :class:`glass.lensing.MultiPlaneConvergence`, which adds a convergence
    plane given by a :class:`~glass.shells.RadialWindow` at its effective
    redshift.

  - The :func:`glass.lensing.multi_plane_matrix` function now takes a sequence
    of :class:`~glass.shells.RadialWindow`.  It no longer returns the list of
    source redshifts, since these are now independently available as the
    effective redshifts of the windows.

- The arguments of the :class:`~glass.lensing.MultiPlaneConvergence` method
  ``add_plane`` have been renamed to ``zsrc`` and ``wlens`` from the more
  ambiguous ``z`` and ``w`` (which could be confused with "window"). The
  properties ``z`` and ``w`` that returned these values have been similarly
  changed.


2023.1 (31 Jan 2023)
--------------------

- **Initial wide release for GLASS paper**

  This was the initial full release of *GLASS*, coinciding with the release of
  preprint `arXiv:2302.01942`__.

  __ https://arxiv.org/abs/2302.01942
