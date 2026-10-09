from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import pytest

pytest.importorskip("jax", reason="tests require jax")

import jax.numpy as jnp
import jax.test_util

import glass

if TYPE_CHECKING:
    from glass._types import FloatArray


def test_broadcast_first() -> None:
    """Tests glass.arraytools.broadcast_first is auto differentiable."""
    a = jnp.ones((2, 3, 4))
    b = jnp.ones((2, 1))

    # arrays with shape ((3, 4, 2)) and ((1, 2)) are passed
    # to xp.broadcast_arrays; hence it works
    jax.test_util.check_grads(
        glass.arraytools.broadcast_first,
        (a, b),
        order=1,
    )


def test_broadcast_leading_axes() -> None:
    """Tests glass.arraytools.broadcast_leading_axes is auto differentiable."""
    a_in = 0.0
    b_in = jnp.zeros((4, 10), dtype=jnp.float64)
    c_in = jnp.zeros((3, 1, 5, 6), dtype=jnp.float64)

    def broadcast_leading_axes_by_arrays(
        a_in: float | FloatArray,
        b_in: float | FloatArray,
        c_in: float | FloatArray,
    ) -> tuple[FloatArray, ...]:
        _, a_out, b_out, c_out = glass.arraytools.broadcast_leading_axes(
            (a_in, 0),
            (b_in, 1),
            (c_in, 2),
        )
        return a_out, b_out, c_out

    jax.test_util.check_grads(
        broadcast_leading_axes_by_arrays,
        (a_in, b_in, c_in),
        order=1,
    )


@pytest.mark.parametrize("axis", [-1, 1])
@pytest.mark.parametrize("left", [0.0, None])
@pytest.mark.parametrize("right", [0.0, None])
def test_ndinterp_without_period(
    axis: int,
    left: float | None,
    right: float | None,
) -> None:
    """Tests glass.arraytools.broadcast_leading_axes is auto differentiable."""
    xq = jnp.asarray([0.0, 1.0, 2.0, 3.0, 4.0])
    yq = jnp.asarray([[1.1, 1.2, 1.3, 1.4, 1.5], [2.1, 2.2, 2.3, 2.4, 2.5]])
    x = jnp.asarray([[0.5, 1.5, 2.5, 3.5], [3.5, 2.5, 1.5, 0.5], [0.5, 3.5, 1.5, 2.5]])

    jax.test_util.check_grads(
        partial(
            glass.arraytools.ndinterp,
            axis=axis,
            left=left,
            right=right,
        ),
        (x, xq, yq),
        order=1,
    )


@pytest.mark.parametrize("axis", [-1, 1])
def test_ndinterp_with_period(
    axis: int,
) -> None:
    """Tests glass.arraytools.ndinterp is auto differentiable."""
    pytest.skip(
        "glass.arraytools.ndinterp is not auto differentiable when period is specified"
    )

    xq = jnp.asarray([0.0, 1.0, 2.0, 3.0, 4.0])
    yq = jnp.asarray([[1.1, 1.2, 1.3, 1.4, 1.5], [2.1, 2.2, 2.3, 2.4, 2.5]])
    x = jnp.asarray([[0.5, 1.5, 2.5, 3.5], [3.5, 2.5, 1.5, 0.5], [0.5, 3.5, 1.5, 2.5]])

    jax.test_util.check_grads(
        partial(
            glass.arraytools.ndinterp,
            axis=axis,
            period=3.5,
        ),
        (x, xq, yq),
        order=1,
    )


def test_trapezoid_product() -> None:
    """Tests glass.arraytools.trapezoid_product is auto differentiable."""
    pytest.skip(
        "glass.arraytools.trapezoid_product is not auto differentiable when period is"
        "specified"
    )

    x1 = jnp.linspace(0, 2, 100)
    f1 = jnp.full_like(x1, 2.0)
    x2 = jnp.linspace(1, 2, 10)
    f2 = jnp.full_like(x2, 0.5)

    jax.test_util.check_grads(
        glass.arraytools.trapezoid_product,
        ((x1, f1), (x2, f2)),
        order=1,
    )


@pytest.mark.parametrize(
    ("f_in", "x_in"),
    [
        ([1, 2, 3, 4], [0, 1, 2, 3]),
        ([[1, 4, 9, 16], [2, 3, 5, 7]], [0, 1, 2.5, 4]),
    ],
    ids=["1D f", "2D f"],
)
def test_cumulative_trapezoid(f_in: FloatArray, x_in: FloatArray) -> None:
    """Tests glass.arraytools.cumulative_trapezoid is auto differentiable."""
    f = jnp.asarray(f_in, dtype=jnp.float64)
    x = jnp.asarray(x_in, dtype=jnp.float64)

    jax.test_util.check_grads(
        glass.arraytools.cumulative_trapezoid,
        (f, x),
        order=1,
    )
