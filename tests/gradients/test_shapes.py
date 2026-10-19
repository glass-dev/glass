from __future__ import annotations

from typing import TYPE_CHECKING

import jax.test_util
import pytest

import glass
import glass.rng

if TYPE_CHECKING:
    from types import ModuleType

    from glass._types import ComplexArray, FloatArray, UnifiedGenerator


@pytest.mark.parametrize("use_xp", [True, False])
def test_triaxial_axis_ratio(
    jnp: ModuleType,
    rng: UnifiedGenerator,
    use_xp: bool,
) -> None:
    """Tests glass.triaxial_axis_ratio is auto differentiable."""
    sorted_uniform_rnd = jnp.sort(
        rng.uniform(
            0,
            1,
            size=(2, 1_000),
        ),
        axis=0,
    )
    zeta = sorted_uniform_rnd[0, :]
    xi = sorted_uniform_rnd[1, :]

    def triaxial_axis_ratio_same_rng(
        xi: FloatArray,
    ) -> FloatArray:
        fresh_rng = glass.rng.default_rng(xp=jnp)
        return glass.triaxial_axis_ratio(
            zeta,
            jnp.asarray(xi),
            rng=fresh_rng,
            xp=jnp if use_xp else None,
        )

    jax.test_util.check_grads(
        triaxial_axis_ratio_same_rng,
        (xi,),
        order=1,
    )


@pytest.mark.parametrize("use_rng", [True, False])
@pytest.mark.parametrize("use_xp", [True, False])
def test_ellipticity_ryden04(
    jnp: ModuleType,
    use_rng: bool,
    use_xp: bool,
) -> None:
    """Tests glass.ellipticity_ryden04 is auto differentiable."""

    def ellipticity_ryden04_diffable(
        mu: FloatArray,
        sigma: FloatArray,
        gamma: FloatArray,
        sigma_gamma: FloatArray,
    ) -> FloatArray:
        fresh_rng = glass.rng.default_rng(xp=jnp)
        return glass.ellipticity_ryden04(
            jnp.asarray(mu),
            jnp.asarray(sigma),
            jnp.asarray(gamma),
            jnp.asarray(sigma_gamma),
            rng=fresh_rng if use_rng else None,
            xp=jnp if use_xp else None,
        )

    jax.test_util.check_grads(
        ellipticity_ryden04_diffable,
        (
            jnp.asarray([-1.9, -2.9]),
            jnp.asarray([0.89, 1.001]),
            jnp.asarray([[0.2, 0.3], [0.4, 0.5]]),
            jnp.asarray([0.056, 0.067]),
        ),
        order=1,
    )


@pytest.mark.parametrize("use_rng", [True, False])
@pytest.mark.parametrize("use_xp", [True, False])
def test_ellipticity_gaussian(
    jnp: ModuleType,
    use_rng: bool,
    use_xp: bool,
) -> None:
    """Tests glass.ellipticity_gaussian is auto differentiable."""
    pytest.skip("glass.ellipticity_gaussian is not auto differentiable")

    count = jnp.asarray([1_000_000, 1_000_000])
    sigma = jnp.asarray([0.128, 0.256])

    def ellipticity_gaussian_diffable(
        sigma: FloatArray,
    ) -> ComplexArray:
        fresh_rng = glass.rng.default_rng(xp=jnp)
        return glass.ellipticity_gaussian(
            count,
            jnp.asarray(sigma),
            rng=fresh_rng if use_rng else None,
            xp=jnp if use_xp else None,
        )

    jax.test_util.check_grads(
        ellipticity_gaussian_diffable,
        (sigma,),
        order=1,
    )


@pytest.mark.parametrize("use_rng", [True, False])
@pytest.mark.parametrize("use_xp", [True, False])
def test_ellipticity_intnorm(
    jnp: ModuleType,
    use_rng: bool,
    use_xp: bool,
) -> None:
    """Tests glass.ellipticity_intnorm is auto differentiable."""
    count = jnp.asarray([1_000_000, 1_000_000])
    sigma = jnp.asarray([0.128, 0.256])

    def ellipticity_intnorm_diffable(
        sigma: FloatArray,
    ) -> ComplexArray:
        fresh_rng = glass.rng.default_rng(xp=jnp)
        return glass.ellipticity_intnorm(
            count,
            jnp.asarray(sigma),
            rng=fresh_rng if use_rng else None,
            xp=jnp if use_xp else None,
        )

    jax.test_util.check_grads(
        ellipticity_intnorm_diffable,
        (sigma,),
        order=1,
    )


@pytest.mark.parametrize(
    ("varg", "vargamma"),
    [
        (None, None),
        (0.1, None),
        (None, 0.1),
    ],
)
def test_resample_shapes(
    jnp: ModuleType,
    rng: UnifiedGenerator,
    varg: float | None,
    vargamma: float | None,
) -> None:
    """Tests glass.resample_shapes is auto differentiable."""
    pytest.skip("glass.resample_shapes is not auto differentiable")

    n = 1_000_000
    r = jnp.sqrt(rng.uniform(0.0, 1.0, n))
    phi = rng.uniform(0.0, 2 * jnp.pi, n)
    epsilon = r * jnp.exp(1j * phi)

    def resample_shapes_diffable(epsilon: ComplexArray) -> ComplexArray:
        fresh_rng = glass.rng.default_rng(xp=jnp)
        return glass.resample_shapes(
            jnp.asarray(epsilon),
            varg=varg,
            vargamma=vargamma,
            rng=fresh_rng,
        )

    jax.test_util.check_grads(
        resample_shapes_diffable,
        (epsilon,),
        order=1,
    )
