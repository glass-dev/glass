from __future__ import annotations

import math
from functools import partial
from typing import TYPE_CHECKING

import jax.test_util
import pytest

import glass

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType
    from typing import Any

    from pytest_mock import MockerFixture

    from glass._types import ComplexArray, FloatArray, UnifiedGenerator


def test_effective_bias(
    jnp: ModuleType,
    mocker: MockerFixture,
) -> None:
    """Tests glass.effective_bias is auto differentiable."""
    w = mocker.Mock()
    w.za = jnp.linspace(0, 2, 100)
    w.wa = jnp.full_like(w.za, 2.0)

    z = jnp.linspace(0, 1, 10)
    bz = jnp.full_like(z, 0.5)

    def effective_bias_diffable(
        bz: FloatArray,
    ) -> float | FloatArray:
        return glass.effective_bias(
            z,
            jnp.asarray(bz),
            w,
        )

    jax.test_util.check_grads(
        effective_bias_diffable,
        (bz,),
        order=1,
    )


def test_linear_bias(
    rng: UnifiedGenerator,
) -> None:
    """Tests glass.linear_bias is auto differentiable."""
    delta = rng.normal(5, 1, size=(2, 2))
    b = 2.0

    jax.test_util.check_grads(
        glass.linear_bias,
        (delta, b),
        order=1,
    )


def test_loglinear_bias(
    rng: UnifiedGenerator,
) -> None:
    """Tests glass.loglinear_bias is auto differentiable."""
    delta = rng.normal(5, 1, size=(2, 2))
    b = 2.0

    jax.test_util.check_grads(
        glass.loglinear_bias,
        (delta, b),
        order=1,
    )


@pytest.mark.parametrize(
    "bias_model",
    [
        glass.linear_bias,
        glass.loglinear_bias,
    ],
)
def test_compute_density_contrast(
    bias_model: Callable[..., Any],
    jnp: ModuleType,
) -> None:
    """Tests glass.points._compute_density_contrast is auto differentiable."""
    bias = 0.8 * jnp.ones((3, 2))
    delta = jnp.zeros((3, 2, 12))
    k = (1, 1)

    def _compute_density_contrast_by_bias_and_delta(
        bias: FloatArray,
        delta: FloatArray,
    ) -> FloatArray:
        return glass.points._compute_density_contrast(bias, bias_model, delta, k)

    jax.test_util.check_grads(
        _compute_density_contrast_by_bias_and_delta,
        (bias, delta),
        order=1,
    )


@pytest.mark.parametrize("remove_monopole", [False, True])
def test_compute_expected_count(
    jnp: ModuleType,
    remove_monopole: bool,
) -> None:
    """Tests glass.points._compute_expected_count is auto differentiable."""
    k = (1, 1)
    n_in = jnp.zeros(12)
    ngal = jnp.tile(jnp.asarray([1e-3, 2e-3]), (3, 1))

    def _compute_expected_count_by_n_and_ngal(
        n: FloatArray,
        ngal: FloatArray,
    ) -> FloatArray:
        return glass.points._compute_expected_count(
            k,
            n,
            ngal,
            remove_monopole=remove_monopole,
        )

    jax.test_util.check_grads(
        _compute_expected_count_by_n_and_ngal,
        (n_in, ngal),
        order=1,
    )


def test_apply_visibility(
    jnp: ModuleType,
) -> None:
    """Tests glass.points._apply_visibility is auto differentiable."""
    k = (1, 1)
    n = 24751.77674965 * jnp.ones(12)
    vis = jnp.tile(jnp.repeat(jnp.asarray([0.0, 1.0]), 6), (3, 2, 1))

    jax.test_util.check_grads(
        partial(glass.points._apply_visibility, k),
        (n, vis),
        order=1,
    )


@pytest.mark.parametrize("bshape", [None, (), (100,), (100, 1)])
@pytest.mark.parametrize("cshape", [(100,), (100, 50), (100, 3, 2)])
def test_position_weights(
    bshape: tuple[int, ...] | None,
    cshape: tuple[int, ...],
    rng: UnifiedGenerator,
) -> None:
    """Tests glass.position_weights is auto differentiable."""
    counts = rng.random(cshape)
    bias = None if bshape is None else rng.random(bshape)

    jax.test_util.check_grads(
        glass.position_weights,
        (counts, bias),
        order=1,
    )


@pytest.mark.parametrize(
    "to_complex",
    [
        lambda r: r + 0j,
        lambda r: -r + 0j,
        lambda r: 1j * r,
        lambda r: -1j * r,
    ],
)
def test_displace(
    jnp: ModuleType,
    to_complex: Callable[..., ComplexArray],
) -> None:
    """Tests glass.displace is auto differentiable."""
    d = 5.0  # deg
    r = d / 180 * math.pi

    # displace the origin so everything is easy
    lon0 = jnp.asarray(0.0)
    lat0 = jnp.asarray(0.0)

    jax.test_util.check_grads(
        glass.displace,
        (lon0, lat0, jnp.asarray(to_complex(r))),
        order=1,
    )


def test_displacement(jnp: ModuleType) -> None:
    """Tests glass.displacement is auto differentiable."""
    zero = jnp.asarray(0.0)
    five = jnp.asarray(5.0)
    ninety = jnp.asarray(90.0)

    jax.test_util.check_grads(
        glass.displacement,
        (zero, ninety, ninety * 2, ninety - five),
        order=1,
    )
