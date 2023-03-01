
Release notes
=============

These notes document the changes between individual *GLASS* releases.

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

  - The ``zmin`` and ``zmax`` parameters of `glass.galaxies.redshifts_from_nz`
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
