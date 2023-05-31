Changelog
=========

All notable changes to the project are documented in this file.  The format is
based on [Keep a Changelog](https://keepachangelog.com).


[2023.5]  (31 May 2023)
-----------------------

### Added

- Allow dimensional input to the sampling functions in `glass.points` (#80)
- The `redshifts_from_nz()` function supports `count` arrays (#83)

### Changed

- Position sampling returns counts alongside points (#80)
- `redshifts_from_nz()` no longer returns `gal_pop` (#83)
- Move core functionality that is used by other, user-facing modules into the
  `glass.core` module (#88)

### Removed

- Remove profiling functions (#89)


[2023.2] - 1 Mar 2023
---------------------

### Added

- The `glass.lensing.MultiPlaneConvergence.add_window` method to add a
  convergence plane given by a window function.
- The `glass.shells` module for shell definitions.
- User functions to save and load Cls
- This changelog added to keep track of changes between versions


### Changed

- Instead of an array of shell boundaries and `MatterWeights`, the shells are
  entirely defined by a `RadialWindow` window function.
- The `glass.lensing.multi_plane_matrix` function now takes a sequence of
  window functions.
- The arguments of `glass.lensing.MultiPlaneConvergence.add_plane` are called
  `zsrc` and `wlens` instead of the more ambiguous `z` and `w`. The properties
  `MultiPlaneConvergence.z` and `MultiPlaneConvergence.w` that return these
  values are similarly changed.
- The `glass.points.effective_bias` now takes a single window function as input
  and computes its effective bias parameter.
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


[2023.1] - 31 Jan 2023
----------------------

### Added

- Initial wide release for GLASS paper


[2023.5]: https://github.com/glass-dev/glass/compare/v2023.2...v2023.5
[2023.2]: https://github.com/glass-dev/glass/compare/v2023.1...v2023.2
[2023.1]: https://github.com/glass-dev/glass/releases/tag/v2023.1
