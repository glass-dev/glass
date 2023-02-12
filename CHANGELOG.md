Changelog
=========

All notable changes to the project are documented in this file.  The format is
based on [Keep a Changelog](https://keepachangelog.com).


[Unreleased]
------------

### Added

- The `glass.lensing.MultiPlaneConvergence.add_window` method to add a
  convergence plane given by a window function.
- The `glass.shells` module for shell definitions.
- User functions to save and load Cls
- This changelog added to keep track of changes between versions


### Changed

- The `glass.lensing.multi_plane_matrix` function now takes sequences `zs, ws`
  of window functions, as well as an optional array `zsrcs` of explicit source
  redshifts.
- The arguments of `glass.lensing.MultiPlaneConvergence.add_plane` are called
  `zsrc` and `wlens` instead of the more ambiguous `z` and `w`. The properties
  `MultiPlaneConvergence.z` and `MultiPlaneConvergence.w` that return these
  values are similarly changed.
- The `glass.points.effective_bias` now takes a single window function `z, w`
  as input and computes its effective bias parameter.
- Some type hints added to library functions


### Removed

- The `glass.lensing.multi_plane_weights` function, replaced by the
  `glass.lensing.MultiPlaneConvergence.add_window` method.
- The `glass.galaxies.constant_densities` and `density_from_dndz` functions,
  since densities can now easily be partitioned by window functions for shells.
- The `zmin, zmax` parameters of `glass.galaxies.redshifts_from_nz`, for the
  same reason.
- The `glass.math.restrict_interval` function, as shells are now defined by
  window functions instead of sharp intervals.
- The `glass.matter` module, in favour of the more appropriately-named
  `glass.shells` module.


[2023.1] - 2023-01-31
---------------------

### Added

- Initial wide release for GLASS paper


[Unreleased]: https://github.com/glass-dev/glass/compare/v2023.1...HEAD
[2023.1]: https://github.com/glass-dev/glass/releases/tag/v2023.1
