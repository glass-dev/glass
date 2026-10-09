from __future__ import annotations

import contextlib
import os
from typing import TYPE_CHECKING

import numpy as np
import pytest

import glass.rng
from tests._optional_dependencies import HAVE_JAX

if TYPE_CHECKING:
    from types import ModuleType

with contextlib.suppress(ImportError):
    # only import if jax is available
    import glass.jax


def test_xp_fixture_selects_requested_backend(xp: ModuleType) -> None:
    expected = {
        "numpy": {"numpy"},
        "array_api_strict": {"array_api_strict"},
        "jax": {"jax.numpy"},
        "all": {"numpy", "array_api_strict", "jax.numpy"},
    }[os.environ.get("ARRAY_BACKEND") or "numpy"]
    assert xp.__name__ in expected


def test_default_rng_numpy() -> None:
    rng = glass.rng.default_rng(xp=np)
    assert isinstance(rng, np.random.Generator)


def test_default_rng_jax(jnp: ModuleType) -> None:
    rng = glass.rng.default_rng(xp=jnp)
    assert isinstance(rng, glass.jax.Generator)


def test_default_rng_array_api_strict(ap: ModuleType) -> None:
    rng = glass.rng.default_rng(xp=ap)
    assert isinstance(rng, glass.rng.Generator)


def test_init(ap: ModuleType) -> None:
    rng = glass.rng.Generator(xp=ap)
    assert isinstance(rng, glass.rng.Generator)


def test_init_mix_of_backends_np_array_api_strict(ap: ModuleType) -> None:
    rng = glass.rng.Generator(rng=glass.rng.default_rng(xp=np), xp=ap)
    assert rng.random(1).__array_namespace__().__name__ == "array_api_strict"
    assert rng.poisson(1).__array_namespace__().__name__ == "array_api_strict"
    assert rng.standard_normal(1).__array_namespace__().__name__ == "array_api_strict"
    assert rng.uniform().__array_namespace__().__name__ == "array_api_strict"
    assert (
        rng.multinomial(1, ap.ones(2)).__array_namespace__().__name__
        == "array_api_strict"
    )


@pytest.mark.skipif(not HAVE_JAX, reason="test requires jax")
def test_init_mix_of_backends_jax_np() -> None:
    rng = glass.rng.Generator(rng=glass.jax.Generator(42), xp=np)
    assert rng.random(1).__array_namespace__().__name__ == "numpy"
    assert rng.poisson(1).__array_namespace__().__name__ == "numpy"
    assert rng.standard_normal(1).__array_namespace__().__name__ == "numpy"
    assert rng.uniform().__array_namespace__().__name__ == "numpy"
    assert rng.multinomial(1, np.ones(2)).__array_namespace__().__name__ == "numpy"


def test_random(ap: ModuleType) -> None:
    rng = glass.rng.default_rng(xp=ap)
    rvs = rng.random(size=10_000)
    assert rvs.shape == (10_000,)
    assert ap.min(rvs) >= 0.0
    assert ap.max(rvs) < 1.0
    assert isinstance(rvs, ap._array_object.Array)


def test_normal(ap: ModuleType) -> None:
    rng = glass.rng.default_rng(xp=ap)
    rvs = rng.normal(1, 2, size=10_000)
    assert rvs.shape == (10_000,)
    assert isinstance(rvs, ap._array_object.Array)


def test_standard_normal(ap: ModuleType) -> None:
    rng = glass.rng.default_rng(xp=ap)
    rvs = rng.standard_normal(size=10_000)
    assert rvs.shape == (10_000,)
    assert isinstance(rvs, ap._array_object.Array)


def test_poisson(ap: ModuleType) -> None:
    rng = glass.rng.default_rng(xp=ap)
    rvs = rng.poisson(lam=1, size=10_000)
    assert rvs.shape == (10_000,)
    assert isinstance(rvs, ap._array_object.Array)


def test_uniform(ap: ModuleType) -> None:
    rng = glass.rng.default_rng(xp=ap)
    rvs = rng.uniform(size=10_000)
    assert rvs.shape == (10_000,)
    assert ap.min(rvs) >= 0.0
    assert ap.max(rvs) < 1.0
    assert isinstance(rvs, ap._array_object.Array)
