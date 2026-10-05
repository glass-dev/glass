from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

pytest.importorskip("jax", reason="tests require jax")

import jax.numpy as jnp
import jax.test_util

import glass
import glass.jax

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import NotImplementedType

    from glass._types import AnyArray, UnifiedGenerator


def test_normal(get_rng: Callable[..., UnifiedGenerator]) -> None:
    """Tests that glass.grf.Normal is auto differentiable when using JAX."""
    t = glass.grf.Normal()
    x = get_rng(jnp).standard_normal(10)

    jax.test_util.check_grads(
        t,
        (x, 1.0),
        order=1,
    )


def test_lognormal(get_rng: Callable[..., UnifiedGenerator]) -> None:
    """Tests that glass.grf.Lognormal is auto differentiable when using JAX."""
    rng = get_rng(jnp)
    lam = rng.uniform()
    var = rng.uniform()
    t = glass.grf.Lognormal(lam)
    x = rng.standard_normal(10)

    jax.test_util.check_grads(
        t,
        (x, var),
        order=1,
    )


def test_sqnormal(get_rng: Callable[..., UnifiedGenerator]) -> None:
    """Tests that glass.grf.SquaredNormal is auto differentiable when using JAX."""
    rng = get_rng(jnp)
    lam = rng.uniform()
    var = rng.uniform()
    a = jnp.sqrt(1 - var)
    t = glass.grf.SquaredNormal(a, lam)
    x = rng.standard_normal(10)

    jax.test_util.check_grads(
        t,
        (x, var),
        order=1,
    )


def test_corr_normal(get_rng: Callable[..., UnifiedGenerator]) -> None:
    """Tests that glass.grf.Normal.corr is auto differentiable when using JAX."""
    t1 = glass.grf.Normal()
    t2 = glass.grf.Normal()
    x = get_rng(jnp).random(10)

    def corr_by_x(x: AnyArray) -> AnyArray | NotImplementedType:
        return t1.corr(t2, x)

    jax.test_util.check_grads(
        corr_by_x,
        (x,),
        order=1,
    )


def test_corr_lognormal(get_rng: Callable[..., UnifiedGenerator]) -> None:
    """Tests that glass.grf.Lognormal.corr is auto differentiable when using JAX."""
    rng = get_rng(jnp)
    lam1 = rng.uniform()
    t1 = glass.grf.Lognormal(lam1)
    lam2 = rng.uniform()
    t2_lognormal = glass.grf.Lognormal(lam2)
    t2_normal = glass.grf.Normal()
    x = rng.random(10)

    def corr_by_x_lognormal(x: AnyArray) -> AnyArray | NotImplementedType:
        return t1.corr(t2_lognormal, x)

    jax.test_util.check_grads(
        corr_by_x_lognormal,
        (x,),
        order=1,
    )

    def corr_by_x_normal(x: AnyArray) -> AnyArray | NotImplementedType:
        return t1.corr(t2_normal, x)

    jax.test_util.check_grads(
        corr_by_x_normal,
        (x,),
        order=1,
    )


def test_corr_sqnormal(get_rng: Callable[..., UnifiedGenerator]) -> None:
    """Tests that glass.grf.SquaredNormal.corr is auto differentiable when using JAX."""
    rng = get_rng(jnp)
    lam1, var1 = rng.uniform(size=2)
    a1 = jnp.sqrt(1 - var1)
    t1 = glass.grf.SquaredNormal(a1, lam1)
    lam2, var2 = rng.uniform(size=2)
    a2 = jnp.sqrt(1 - var2)
    t2 = glass.grf.SquaredNormal(a2, lam2)
    x = rng.random(10)

    def corr_by_x(x: AnyArray) -> AnyArray | NotImplementedType:
        return t1.corr(t2, x)

    jax.test_util.check_grads(
        corr_by_x,
        (x,),
        order=1,
    )
