from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import jax.test_util
import pytest

import glass.arraytools

if TYPE_CHECKING:
    from types import ModuleType

    from glass._types import FloatArray


def test_broadcast_first(jnp: ModuleType) -> None:
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


def test_broadcast_leading_axes(jnp: ModuleType) -> None:
    """Tests glass.arraytools.broadcast_leading_axes is auto differentiable."""
    a_in = 0.0
    b_in = jnp.zeros((4, 10))
    c_in = jnp.zeros((3, 1, 5, 6))

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
@pytest.mark.parametrize("period", [4.5, None])
@pytest.mark.parametrize("right", [0.0, None])
def test_ndinterp(
    axis: int,
    jnp: ModuleType,
    left: float | None,
    period: float | None,
    right: float | None,
) -> None:
    """
    Tests glass.arraytools.ndinterp is auto differentiable.

    For glass.arraytools.ndinterp to be auto differentiable, the period must keep the
    sample coordinates away from the wrapping boundary.
    """
    xq = jnp.asarray([0.0, 1.0, 2.0, 3.0, 4.0])
    yq = jnp.asarray([[1.1, 1.2, 1.3, 1.4, 1.5], [2.1, 2.2, 2.3, 2.4, 2.5]])
    x = jnp.asarray([[0.5, 1.5, 2.5, 3.5], [3.5, 2.5, 1.5, 0.5], [0.5, 3.5, 1.5, 2.5]])

    jax.test_util.check_grads(
        functools.partial(
            glass.arraytools.ndinterp,
            axis=axis,
            left=left,
            right=right,
            period=period,
        ),
        (x, xq, yq),
        order=1,
    )


def test_trapezoid_product(jnp: ModuleType) -> None:
    """Tests glass.arraytools.trapezoid_product is auto differentiable."""
    x1 = jnp.linspace(0, 2, 100)
    f1 = jnp.full_like(x1, 2.0)
    x2 = jnp.linspace(1, 2, 10)
    f2 = jnp.full_like(x2, 0.5)

    def trapezoid_product_diffable(
        f1: FloatArray,
        f2: FloatArray,
    ) -> float | FloatArray:
        return glass.arraytools.trapezoid_product(
            (x1, jnp.asarray(f1)),
            (x2, jnp.asarray(f2)),
        )

    jax.test_util.check_grads(
        trapezoid_product_diffable,
        (f1, f2),
        order=1,
    )


@pytest.mark.parametrize(
    ("f_in", "x_in"),
    [
        ([1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 2.0, 3.0]),
        ([[1.0, 4.0, 9.0, 16.0], [2.0, 3.0, 5.0, 7.0]], [0.0, 1.0, 2.5, 4.0]),
    ],
    ids=["1D f", "2D f"],
)
def test_cumulative_trapezoid(
    f_in: FloatArray,
    jnp: ModuleType,
    x_in: FloatArray,
) -> None:
    """Tests glass.arraytools.cumulative_trapezoid is auto differentiable."""
    f = jnp.asarray(f_in)
    x = jnp.asarray(x_in)

    jax.test_util.check_grads(
        glass.arraytools.cumulative_trapezoid,
        (f, x),
        order=1,
    )
