from __future__ import annotations

from typing import TYPE_CHECKING

import jax.test_util
import pytest

import glass

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

    from glass._types import FloatArray
    from glass.cosmology import Cosmology


def test_distance_weight(
    get_cosmo: Callable[..., Cosmology],
    jnp: ModuleType,
) -> None:
    """Tests glass.DistanceWeight is auto differentiable."""
    cosmo = get_cosmo(jnp)
    z = jnp.linspace(0, 1, 6)

    def distance_weight_diffable(z: FloatArray) -> FloatArray:
        return glass.DistanceWeight(cosmo)(z)

    jax.test_util.check_grads(
        distance_weight_diffable,
        (z,),
        order=1,
    )


def test_volume_weight(
    get_cosmo: Callable[..., Cosmology],
    jnp: ModuleType,
) -> None:
    """Tests glass.VolumeWeight is auto differentiable."""
    z = jnp.linspace(0.1, 1, 6, dtype=jnp.float64)
    cosmo = get_cosmo(jnp)

    def volume_weight_diffable(z: FloatArray) -> FloatArray:
        return glass.VolumeWeight(cosmo)(jnp.asarray(z))

    jax.test_util.check_grads(
        volume_weight_diffable,
        (z,),
        order=1,
    )


def test_density_weight(
    get_cosmo_with_omega_m: Callable[..., Cosmology],
    jnp: ModuleType,
) -> None:
    """Tests glass.DensityWeight is auto differentiable."""
    z = jnp.linspace(0.1, 1, 6)
    cosmo = get_cosmo_with_omega_m(jnp)

    def density_weight_diffable(z: FloatArray) -> FloatArray:
        return glass.DensityWeight(cosmo)(z)

    jax.test_util.check_grads(
        density_weight_diffable,
        (z,),
        order=1,
    )


@pytest.mark.filterwarnings(
    "ignore:first tophat window does not start at redshift zero",
)
def test_tophat_windows(jnp: ModuleType) -> None:
    """Tests glass.tophat_windows is auto differentiable."""
    pytest.skip("glass.tophat_windows is not auto differentiable")

    zb = jnp.asarray([0.0, 0.1, 0.2, 0.5, 1.0, 2.0])

    def tophat_windows_diffable(zb: FloatArray) -> list[tuple[FloatArray, FloatArray]]:
        dz = 5e-3
        ws = glass.tophat_windows(zb, dz)
        return [(w.za, w.wa) for w in ws]

    jax.test_util.check_grads(
        tophat_windows_diffable,
        (zb,),
        order=1,
    )


@pytest.mark.filterwarnings("ignore:first triangular window does not start at z=0")
@pytest.mark.parametrize("weight", [None, lambda x: 2 * x])
def test_linear_windows(
    jnp: ModuleType,
    weight: Callable[[FloatArray], FloatArray] | None,
) -> None:
    """Tests glass.linear_windows is auto differentiable."""
    pytest.skip("glass.linear_windows is not auto differentiable")

    zgrid = jnp.asarray(
        [
            0.0,
            0.20224358,
            0.42896272,
            0.69026819,
            1.0,
        ],
    )

    def linear_windows_diffable(
        zgrid: FloatArray,
    ) -> list[tuple[FloatArray, FloatArray]]:
        dz = 1e-2
        ws = glass.linear_windows(zgrid, dz, weight=weight)
        return [(w.za, w.wa) for w in ws]

    jax.test_util.check_grads(
        linear_windows_diffable,
        (zgrid,),
        order=1,
    )


@pytest.mark.filterwarnings("ignore:first cubic spline window does not start at z=0")
@pytest.mark.parametrize("weight", [None, lambda x: 2 * x])
def test_cubic_windows(
    jnp: ModuleType,
    weight: Callable[[FloatArray], FloatArray] | None,
) -> None:
    """Tests glass.cubic_windows is auto differentiable."""
    pytest.skip("glass.cubic_windows is not auto differentiable")

    zgrid = jnp.asarray(
        [
            0.0,
            0.20224358,
            0.42896272,
            0.69026819,
            1.0,
        ],
    )

    def cubic_windows_diffable(
        zgrid: FloatArray,
    ) -> list[tuple[FloatArray, FloatArray]]:
        dz = 1e-2
        ws = glass.cubic_windows(zgrid, dz, weight=weight)
        return [(w.za, w.wa) for w in ws]

    jax.test_util.check_grads(
        cubic_windows_diffable,
        (zgrid,),
        order=1,
    )


def test_restrict(
    jnp: ModuleType,
) -> None:
    """Tests glass.restrict is auto differentiable."""
    z = jnp.linspace(0.1, 5.0, 1_000)
    f = jnp.exp(-(((z - 2.0) / 0.5) ** 2) / 2)

    w = glass.RadialWindow(
        za=jnp.asarray([1.0, 2.0, 3.0, 4.0]),
        wa=jnp.asarray([0.0, 0.5, 0.5, 0.0]),
    )

    def restrict_diffable(
        f: FloatArray,
    ) -> tuple[FloatArray, FloatArray]:
        zr, fr = glass.restrict(
            jnp.asarray(z),
            jnp.asarray(f),
            w,
        )
        return zr, fr

    jax.test_util.check_grads(
        restrict_diffable,
        (f,),
        order=1,
    )


@pytest.mark.parametrize("method", ["lstsq", "nnls", "restrict"])
def test_partition(
    method: str,
    get_shells: Callable[..., list[glass.RadialWindow]],
    jnp: ModuleType,
) -> None:
    """Tests glass.partition is auto differentiable."""
    z = jnp.linspace(0.0, 5.0, 1_000)
    k = 1.0 + jnp.reshape(jnp.arange(6.0), (3, 2, 1))
    fz = jnp.exp(-z / k)
    shells = get_shells(jnp)

    def partition_diffable(fz: FloatArray) -> FloatArray:
        return glass.partition(
            z,
            jnp.asarray(fz),
            shells,
            method=method,
        )

    jax.test_util.check_grads(
        partition_diffable,
        (fz,),
        order=1,
    )


def test_combine(
    get_shells: Callable[..., list[glass.RadialWindow]],
    jnp: ModuleType,
) -> None:
    """Tests glass.combine is auto differentiable."""
    z = jnp.linspace(0.0, 5.0, 1_000)
    weights = jnp.asarray(
        [1.0, 0.90595172, 0.81025465, 0.72003963, 0.63892872, 0.56796183],
    )
    shells = get_shells(jnp)

    def combine_diffable(
        weights: FloatArray,
    ) -> FloatArray:
        return glass.combine(
            z,
            jnp.asarray(weights),
            shells,
        )

    jax.test_util.check_grads(
        combine_diffable,
        (weights,),
        order=1,
    )
