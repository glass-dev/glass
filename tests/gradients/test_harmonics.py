from __future__ import annotations

from typing import TYPE_CHECKING

import jax.test_util
import pytest

import glass.harmonics

if TYPE_CHECKING:
    from types import ModuleType

    from glass._types import ComplexArray, FloatArray


def test_multalm(jnp: ModuleType) -> None:
    """Tests glass.harmonics.multalm is auto differentiable using JAX."""
    alm = jnp.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    bl = jnp.asarray([2.0, 0.5, 1.0])

    jax.test_util.check_grads(glass.harmonics.multalm, (alm, bl), order=1)


@pytest.mark.parametrize("spin", [0, 1, 2])
@pytest.mark.parametrize("pixwin", [True, False])
@pytest.mark.parametrize("pol", [True, False])
def test_inverse_transform(
    jnp: ModuleType,
    spin: int,
    pixwin: bool,
    pol: bool,
) -> None:
    """Tests glass.harmonics.inverse_transform is auto differentiable using JAX."""
    pytest.skip("glass.harmonics.inverse_transform is not auto differentiable")

    alm = jnp.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    lmax = 2
    nside = 1

    def inverse_transform_diffable(
        alm: ComplexArray,
    ) -> FloatArray | ComplexArray:
        return glass.harmonics.inverse_transform(
            jnp.asarray(alm),
            lmax=lmax,
            nside=nside,
            pixwin=pixwin,
            pol=pol,
            spin=spin,
        )

    jax.test_util.check_grads(
        inverse_transform_diffable,
        (alm,),
        order=1,
    )


@pytest.mark.parametrize("pol", [True, False])
@pytest.mark.parametrize("use_pixel_weights", [True, False])
def test_transform(
    jnp: ModuleType,
    pol: bool,
    use_pixel_weights: bool,
) -> None:
    """Tests glass.harmonics.transform is auto differentiable using JAX."""
    pytest.skip("glass.harmonics.transform is not auto differentiable")

    maps = jnp.asarray([1.0] * 12)
    lmax = 2

    def transform_diffable(
        maps: FloatArray,
    ) -> ComplexArray:
        return glass.harmonics.transform(
            jnp.asarray(maps),
            lmax=lmax,
            pol=pol,
            use_pixel_weights=use_pixel_weights,
        )

    jax.test_util.check_grads(
        transform_diffable,
        (maps,),
        order=1,
    )
