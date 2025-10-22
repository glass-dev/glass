<!-- markdownlint-disable MD024 -->

# Changelog

All functional changes to the project are documented in this file.

## [2025.2] (21 Oct 2025)

- gh-578: mutable argument should be empty list rather than `None` (#579)
- gh-552: add docs on all-contributors (#565)
- gh-573: add keyword `rng` to functions in notebooks (#574)
- gh-583: fix flakiness of `test_generate` (#584)
- gh-575: bring back inv_triangle_number (#577)
- gh-572: make `multalm` private (#576)
- gh-487: reorder alms in \_generate_grf to glass ordering (#533)
- gh-568: convert `_weight` funcs to classes for `_windows` funcs (#589)
- gh-495: allow `zeff` to be computed automatically (#590)
- gh-566: add example on galaxy redshift distributions (#567)
- gh-616: use `position_weights()` in galaxy redshift example (#617)
- gh-124: use `cosmology.api` over the `cosmology` package (#537)
- gh-621: bump `array-api-strict` version to `2024.12` (#625)
- gh-639: fix compute_gaussian_spectra() for empty spectra (#640)
- gh-652: make `array-api-strict` an official dependency (#653)
- gh-407: Port functions in fields.py (#642)
- gh-633: switch to `dependency-groups` (#634)
- gh-408: porting straightforward functions in `shells` (#643)
- gh-667: Make JAX an optional dependency as intended (#672)
- gh-409: Port straightforward functions in `points` (#663)
- gh-691: rename envvar to `ARRAY_BACKEND` (#692)
- gh-693: Add script to generate release notes (#694)

## [2025.1] (21 Feb 2025)

- gh-163: drop `Python 3.9` & `NumPy < 2.1.1` support (#391)
- gh-368: add type overloading for `lensing.from_convergence` (#395)
- gh-384: rename all internal functions to `trapezoid` over `trapz` (#392)
- gh-440: remove scipy as a test dependency (#462)
- gh-443: `if` should be `elif` in `fixed_zbins` (#444)
- gh-445: place all public function in `__all__` (#446)
- gh-465: enable installing glass through git archives (#483)
- gh-470: instructions to install using conda (#481)
- gh-478: `glass.core.array` -> `glass.arraytools` (#524)
- gh-493: add module for Gaussian random fields (#494)
- gh-496: functions for legacy mode (#497)
- gh-501: change imports to be directly from `glass` (#512)
- gh-504: deprecate old generator functions (#523)
- gh-508: create standalone bibliography page (#518)
- gh-509: `inv_triangle_number` -> `nfields_from_nspectra` (plus make it public)
  (#527)
- gh-513: make algorithm a public module (#514)

## [2024.2] (15 Nov 2024)

- gh-188: add docstrings to all functions and tidy docs (#381)
- gh-336: support Python 3.13 (#337)
- gh-358: add static types support (#368)
- gh-131: rename `gaussian_gls` to `discretized_cls` (#345)
- gh-328: efficient resampling in `ellipticity_ryden04` (#341)
- gh-137: deprecate `redshifts_from_nz` in favor of `redshifts` (#333)
- gh-328: fix shape mismatch bug in ellipticity_ryden04 (#332)
- gh-315: add broadcasting rule in ellipticity_ryden04 + tests (#317)
- gh-198: enforce `python>3.8` & `numpy>1.21` (#326)
- gh-260: remove glass.core.constants (#261)
- gh-107: add all public functions/classes under glass namespace (#221)
- gh-168: move examples into repository (#169)
- gh-156: add FITS catalogue writer tool (#158)

## [2024.1] (16 Jul 2024)

### Added

- A new function `combine()` that evaluates the linear combination of radial
  window functions with given weights.
- A new function `effective_cls()` which combines power spectra using a list of
  weights, which models what happens in the simulation.
- A new function `position_weights()` that returns weights for `effective_cls()`
  to model the result of `positions_from_delta()`.
- A new function `multi_plane_weights()` that returns weights for
  `effective_cls()` to model the result of `MultiPlaneConvergence`.
- The `glass.core.algorithm` module.
- The new `partition(method="nnls")` function computes a partition with
  non-negative contributions for each shell.
- Function `redshifts()` to sample redshifts following a radial window function.

### Changed

- The default method for `partition()` is now `"nnls"`.
- Both `partition(method="nnls")` and `partition(method="lstsq")` now have an
  additional integral constraint so that the sum of the partition recovers the
  integral of the input function.
- The output of `partition()` now has the shells axis as its first.

### Fixed

- Now uses the updated intersphinx URL for the GLASS examples.
- A bug in `effective_cls()` that caused arrays to be one entry too long if
  `lmax` was not given explicitly.
- A bug in `partition()` with the default method.
- `partition()` now works correctly with functions having extra axes.

## [2023.7] (1 Aug 2023)

### Added

- Function `getcl()` to return angular power spectra by index from a list using
  GLASS ordering.
- New `linear_windows()` and `cubic_windows()` window functions for shells.

### Changed

- The `gaussian_phz()` function now accepts bounds using `lower=` and `upper=`
  keyword parameters.
- The `partition()` function now returns an array of weights to approximate the
  given function by the windows.

## [2023.6] (30 Jun 2023)

### Added

- `deflect()` applies deflections to positions
- `from_convergence()` returns other lensing fields given the convergence
- A new `glass.ext` namespace, reserved for extensions

### Changed

- The `glass` module is no longer a namespace package
- The point sampling functions `positions_from_delta()` and
  `uniform_positions()` now return an iterator
- `ellipticity_gaussian()` and `ellipticity_intnorm()` accept array inputs
- Use pyproject.toml for packaging

### Deprecated

- `shear_from_convergence()` is deprecated in favour of `from_convergence()`

### Removed

- The `glass.all` meta-module is no longer necessary

### Fixed

- Incorrect extrapolation in `glass.core.array.trapz_product()`, causing a bug
  in `glass.points.effective_bias()`

## [2023.5] (31 May 2023)

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

## [2023.2] - 1 Mar 2023

### Added

- The `glass.lensing.MultiPlaneConvergence.add_window` method to add a
  convergence plane given by a window function.
- The `glass.shells` module for shell definitions.
- User functions to save and load Cls
- This changelog added to keep track of changes between versions

### Changed

- Instead of an array of shell boundaries and `MatterWeights`, the shells are
  entirely defined by a `RadialWindow` window function.
- The `glass.lensing.multi_plane_matrix` function now takes a sequence of window
  functions.
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

## [2023.1] - 31 Jan 2023

### Added

- Initial wide release for GLASS paper

[2025.2]: https://github.com/glass-dev/glass/compare/v2025.1...v2025.2
[2025.1]: https://github.com/glass-dev/glass/compare/v2024.2...v2025.1
[2024.2]: https://github.com/glass-dev/glass/compare/v2024.1...v2024.2
[2024.1]: https://github.com/glass-dev/glass/compare/v2023.7...v2024.1
[2023.7]: https://github.com/glass-dev/glass/compare/v2023.6...v2023.7
[2023.6]: https://github.com/glass-dev/glass/compare/v2023.5...v2023.6
[2023.5]: https://github.com/glass-dev/glass/compare/v2023.2...v2023.5
[2023.2]: https://github.com/glass-dev/glass/compare/v2023.1...v2023.2
[2023.1]: https://github.com/glass-dev/glass/releases/tag/v2023.1
