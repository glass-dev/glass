from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

jax = pytest.importorskip("jax", reason="tests require jax")
import jax.test_util

import glass
from glass import rng

if TYPE_CHECKING:
    from glass._types import AngularPowerSpectra, UnifiedGenerator

jnp = jax.numpy

@pytest.fixture(scope="session")
def urng() -> UnifiedGenerator:
    return rng.default_rng(xp=jnp)

def test_normal(urng: UnifiedGenerator) -> None:
    """Tests that glass.grf.Normal is auto differentiable when using JAX."""
    t = glass.grf.Normal()
    urng = rng.default_rng(xp=jnp)
    x = urng.standard_normal(10)

    jax.test_util.check_grads(
        t,
        (x, 1.0),
        order=1,
    )


def test_lognormal(urng: UnifiedGenerator) -> None:
    """Tests that glass.grf.Lognormal is auto differentiable when using JAX."""
    lam = urng.uniform()
    var = urng.uniform()
    t = glass.grf.Lognormal(lam)
    x = urng.standard_normal(10)
    y = lam * jnp.expm1(x - var / 2)

    jax.test_util.check_grads(
        t,
        (x, var),
        order=1,
    )


def test_sqnormal(urng: UnifiedGenerator) -> None:
    """Tests that glass.grf.SquaredNormal is auto differentiable when using JAX."""
    lam = urng.uniform()
    var = urng.uniform()
    a = jnp.sqrt(1 - var)
    t = glass.grf.SquaredNormal(a, lam)
    x = urng.standard_normal(10)
    y = lam * ((x - a) ** 2 - 1)

    jax.test_util.check_grads(
        t,
        (x, var),
        order=1,
    )


def test_corr_normal_normal(urng: UnifiedGenerator) -> None:
    """Tests that glass.grf.Normal.corr is auto differentiable when using JAX and Normal."""
    t1 = glass.grf.Normal()
    t2 = glass.grf.Normal()
    x = urng.random(10)

    def corr_by_x(x):
        return t1.corr(t2, x)

    jax.test_util.check_grads(
        corr_by_x,
        (x,),
        order=1,
    )


def test_corr_lognormal_lognormal(urng: UnifiedGenerator) -> None:
    """Tests that glass.grf.Lognormal.corr is auto differentiable when using JAX with Lognormal."""
    lam1 = urng.uniform()
    t1 = glass.grf.Lognormal(lam1)
    lam2 = urng.uniform()
    t2 = glass.grf.Lognormal(lam2)
    x = urng.random(10)

    def corr_by_x(x):
        return t1.corr(t2, x)

    jax.test_util.check_grads(
        corr_by_x,
        (x,),
        order=1,
    )


def test_corr_lognormal_normal(urng: UnifiedGenerator) -> None:
    """Tests that glass.grf.Lognormal.corr is auto differentiable when using JAX with Normal."""
    lam1 = urng.uniform()
    t1 = glass.grf.Lognormal(lam1)
    t2 = glass.grf.Normal()
    x = urng.random(10)

    def corr_by_x(x):
        return t1.corr(t2, x)

    jax.test_util.check_grads(
        corr_by_x,
        (x,),
        order=1,
    )


def test_corr_sqnormal_sqnormal(urng: UnifiedGenerator) -> None:
    """Tests that glass.grf.SquaredNormal.corr is auto differentiable when using JAX with SquaredNormal."""
    lam1, var1 = urng.uniform(size=2)
    a1 = jnp.sqrt(1 - var1)
    t1 = glass.grf.SquaredNormal(a1, lam1)
    lam2, var2 = urng.uniform(size=2)
    a2 = jnp.sqrt(1 - var2)
    t2 = glass.grf.SquaredNormal(a2, lam2)
    x = urng.random(10)

    def corr_by_x(x):
        return t1.corr(t2, x)

    jax.test_util.check_grads(
        corr_by_x,
        (x,),
        order=1,
    )
