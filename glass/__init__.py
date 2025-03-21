"""GLASS package."""

__all__ = [
    "DensityWeight",
    "DistanceWeight",
    "MultiPlaneConvergence",
    "RadialWindow",
    "VolumeWeight",
    "algorithm",
    "check_posdef_spectra",
    "cls2cov",
    "combine",
    "compute_gaussian_spectra",
    "cov_from_spectra",
    "cubic_windows",
    "deflect",
    "discretized_cls",
    "distance_grid",
    "effective_bias",
    "effective_cls",
    "ellipticity_gaussian",
    "ellipticity_intnorm",
    "ellipticity_ryden04",
    "enumerate_spectra",
    "equal_dens_zbins",
    "fixed_zbins",
    "from_convergence",
    "galaxy_shear",
    "gaussian_fields",
    "gaussian_nz",
    "gaussian_phz",
    "generate",
    "generate_gaussian",
    "generate_lognormal",
    "getcl",
    "glass_to_healpix_spectra",
    "grf",
    "healpix_to_glass_spectra",
    "iternorm",
    "linear_bias",
    "linear_windows",
    "load_cls",
    "loglinear_bias",
    "lognormal_fields",
    "lognormal_gls",
    "lognormal_shift_hilbert2011",
    "multi_plane_matrix",
    "multi_plane_weights",
    "nfields_from_nspectra",
    "partition",
    "position_weights",
    "positions_from_delta",
    "redshift_grid",
    "redshifts",
    "redshifts_from_nz",
    "regularized_spectra",
    "restrict",
    "save_cls",
    "shear_from_convergence",
    "smail_nz",
    "solve_gaussian_spectra",
    "spectra_indices",
    "tomo_nz_gausserr",
    "tophat_windows",
    "triaxial_axis_ratio",
    "uniform_positions",
    "vmap_galactic_ecliptic",
    "write_catalog",
]


try:  # noqa: SIM105
    from ._version import __version__
except ModuleNotFoundError:
    pass

# modules
from glass import grf
from glass.fields import (
    check_posdef_spectra,
    cls2cov,
    compute_gaussian_spectra,
    cov_from_spectra,
    discretized_cls,
    effective_cls,
    enumerate_spectra,
    gaussian_fields,
    generate,
    generate_gaussian,
    generate_lognormal,
    getcl,
    glass_to_healpix_spectra,
    healpix_to_glass_spectra,
    iternorm,
    lognormal_fields,
    lognormal_gls,
    lognormal_shift_hilbert2011,
    nfields_from_nspectra,
    regularized_spectra,
    solve_gaussian_spectra,
    spectra_indices,
)
from glass.galaxies import (
    galaxy_shear,
    gaussian_phz,
    redshifts,
    redshifts_from_nz,
)
from glass.lensing import (
    MultiPlaneConvergence,
    deflect,
    from_convergence,
    multi_plane_matrix,
    multi_plane_weights,
    shear_from_convergence,
)
from glass.observations import (
    equal_dens_zbins,
    fixed_zbins,
    gaussian_nz,
    smail_nz,
    tomo_nz_gausserr,
    vmap_galactic_ecliptic,
)
from glass.points import (
    effective_bias,
    linear_bias,
    loglinear_bias,
    position_weights,
    positions_from_delta,
    uniform_positions,
)
from glass.shapes import (
    ellipticity_gaussian,
    ellipticity_intnorm,
    ellipticity_ryden04,
    triaxial_axis_ratio,
)
from glass.shells import (
    DensityWeight,
    DistanceWeight,
    RadialWindow,
    VolumeWeight,
    combine,
    cubic_windows,
    distance_grid,
    linear_windows,
    partition,
    redshift_grid,
    restrict,
    tophat_windows,
)
from glass.user import load_cls, save_cls, write_catalog
