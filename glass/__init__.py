"""GLASS package."""

import contextlib
from importlib.metadata import PackageNotFoundError

with contextlib.suppress(PackageNotFoundError):
    from ._version import __version__, __version_tuple__

from glass.fields import (
    cls2cov,
    discretized_cls,
    effective_cls,
    generate_gaussian,
    generate_lognormal,
    getcl,
    iternorm,
    lognormal_gls,
    multalm,
    transform_cls,
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
    RadialWindow,
    combine,
    cubic_windows,
    density_weight,
    distance_grid,
    distance_weight,
    linear_windows,
    partition,
    redshift_grid,
    restrict,
    tophat_windows,
    volume_weight,
)
from glass.user import load_cls, save_cls, write_catalog

__all__ = [
    "cls2cov",
    "combine",
    "cubic_windows",
    "deflect",
    "density_weight",
    "discretized_cls",
    "distance_grid",
    "distance_weight",
    "effective_bias",
    "effective_cls",
    "ellipticity_gaussian",
    "ellipticity_intnorm",
    "ellipticity_ryden04",
    "equal_dens_zbins",
    "fixed_zbins",
    "from_convergence",
    "galaxy_shear",
    "gaussian_nz",
    "gaussian_phz",
    "generate_gaussian",
    "generate_lognormal",
    "getcl",
    "iternorm",
    "linear_bias",
    "linear_windows",
    "load_cls",
    "loglinear_bias",
    "lognormal_gls",
    "multalm",
    "multi_plane_matrix",
    "multi_plane_weights",
    "MultiPlaneConvergence",
    "partition",
    "position_weights",
    "positions_from_delta",
    "RadialWindow",
    "redshift_grid",
    "redshifts",
    "redshifts_from_nz",
    "restrict",
    "save_cls",
    "shear_from_convergence",
    "smail_nz",
    "tomo_nz_gausserr",
    "tophat_windows",
    "transform_cls",
    "triaxial_axis_ratio",
    "uniform_positions",
    "vmap_galactic_ecliptic",
    "volume_weight",
    "write_catalog",
]
