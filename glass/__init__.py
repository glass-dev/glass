from importlib.metadata import PackageNotFoundError

try:
    from ._version import __version__, __version_tuple__
except PackageNotFoundError:
    pass

from glass.fields import (
    iternorm,
    cls2cov,
    multalm,
    transform_cls,
    gaussian_gls,
    lognormal_gls,
    generate_gaussian,
    generate_lognormal,
    getcl,
    effective_cls,
)
from glass.galaxies import (
    redshifts,
    redshifts_from_nz,
    galaxy_shear,
    gaussian_phz,
)
from glass.lensing import (
    from_convergence,
    shear_from_convergence,
    MultiPlaneConvergence,
    multi_plane_matrix,
    multi_plane_weights,
    deflect,
)
from glass.observations import (
    vmap_galactic_ecliptic,
    gaussian_nz,
    smail_nz,
    fixed_zbins,
    equal_dens_zbins,
    tomo_nz_gausserr,
)
from glass.points import (
    effective_bias,
    linear_bias,
    loglinear_bias,
    positions_from_delta,
    uniform_positions,
    position_weights,
)
from glass.shapes import (
    triaxial_axis_ratio,
    ellipticity_ryden04,
    ellipticity_gaussian,
    ellipticity_intnorm,
)
from glass.shells import (
    distance_weight,
    volume_weight,
    density_weight,
    tophat_windows,
    linear_windows,
    cubic_windows,
    restrict,
    partition,
    redshift_grid,
    distance_grid,
    combine,
    RadialWindow,
)
from glass.user import save_cls, load_cls, write_catalog
