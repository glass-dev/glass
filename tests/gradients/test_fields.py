from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import pytest

pytest.importorskip("jax", reason="tests require jax")

import jax.numpy as jnp
import jax.test_util

import glass

if TYPE_CHECKING:
    from glass._types import AngularPowerSpectra, AnyArray, FloatArray, UnifiedGenerator


@pytest.mark.parametrize("k", [0, 1, 2])
@pytest.mark.parametrize("test_nd", [False, True])
def test_iternorm(
    k: int,
    test_nd: bool,
) -> None:
    """Tests glass.iternorm is auto differentiable when using JAX."""
    # covariance matrix to sample
    if test_nd:
        cov = jnp.asarray(
            [
                # first cov
                [
                    [1.0, 0.2, 0.1],
                    [0.2, 0.5, 0.2],
                    [0.1, 0.2, 0.3],
                ],
                # second cov
                [
                    [1.4, 0.4, 0.5],
                    [0.4, 1.5, 0.8],
                    [0.5, 0.8, 1.3],
                ],
            ],
        )
    else:
        cov = jnp.asarray(
            [
                [1.0, 0.2, 0.1],
                [0.2, 0.5, 0.2],
                [0.1, 0.2, 0.3],
            ],
        )

    cov = [cov[..., i, i::-1][..., : min(i, k) + 1] for i in range(cov.shape[-1])]

    def iternorm_no_generator(cov: list[FloatArray]) -> list[FloatArray]:
        return list(glass.iternorm(cov))

    jax.test_util.check_grads(
        iternorm_no_generator,
        (cov,),
        order=1,
    )


def test_cls2cov() -> None:
    """Tests glass.cls2cov is auto differentiable when using JAX."""
    pytest.skip("glass.cls2cov is not auto differentiable")

    def cls2cov_no_generator(
        cls: AngularPowerSpectra,
    ) -> list[FloatArray]:
        return list(partial(glass.cls2cov, nl=3, nf=3, nc=2)(cls))

    jax.test_util.check_grads(
        cls2cov_no_generator,
        (
            [
                jnp.asarray(arr)
                for arr in [
                    [1.0, 0.5, 0.3],
                    [0.8, 0.4, 0.2],
                    [0.7, 0.6, 0.1],
                    [0.9, 0.5, 0.3],
                    [0.6, 0.3, 0.2],
                    [0.8, 0.7, 0.4],
                ]
            ],
        ),
        order=1,
    )


@pytest.mark.parametrize("lmax", [None, 5])
@pytest.mark.parametrize("ncorr", [None, 0])
@pytest.mark.parametrize("nside", [None, 4])
def test_discretized_cls(
    lmax: int | None,
    ncorr: int | None,
    nside: int | None,
) -> None:
    """Tests glass.discretized_cls is auto differentiable when using JAX."""
    pytest.skip("glass.discretized_cls is not auto differentiable")

    cls = [jnp.arange(10) for i in range(3)]

    jax.test_util.check_grads(
        partial(glass.discretized_cls, lmax=lmax, ncorr=ncorr, nside=nside),
        (cls,),
        order=1,
    )


@pytest.mark.parametrize("use_rng", [False, True])
@pytest.mark.parametrize("ncorr", [None, 1])
def test_generate_grf(
    ncorr: int | None,
    use_rng: bool,
) -> None:
    """Tests glass.fields._generate_grf is auto differentiable when using JAX."""
    pytest.skip("glass.fields._generate_grf is not auto differentiable")

    gls: AngularPowerSpectra = [jnp.asarray([1.0, 0.5, 0.1])]

    def _generate_grf_by_cls(
        gls: AnyArray,
        # nside: int,
    ) -> list[FloatArray]:
        # Must supply a fresh JAX RNG so that we generate the same sequence for each
        # call carried out by jax.test_util.check_grads
        rng = glass.rng.default_rng(xp=jnp) if use_rng else None
        return list(
            glass.fields._generate_grf(
                gls,
                nside=4,
                rng=rng,
                ncorr=ncorr,
            )
        )

    jax.test_util.check_grads(
        _generate_grf_by_cls,
        (gls,),
        order=1,
    )


@pytest.mark.parametrize(("i", "j"), [(0, 9), (5, 5)])
@pytest.mark.parametrize("lmax", [50, 0, None])
def test_getcl(
    i: int,
    j: int,
    lmax: int | None,
) -> None:
    """Tests glass.getcl is auto differentiable when using JAX."""
    cls: AngularPowerSpectra = [
        jnp.asarray([i, j], dtype=jnp.float64)
        for i in range(10)
        for j in range(i, -1, -1)
    ]

    jax.test_util.check_grads(
        partial(glass.getcl, i=i, j=j, lmax=lmax),
        (cls,),
        order=1,
    )


def test_enumerate_spectra() -> None:
    """Tests glass.enumerate_spectra is auto differentiable when using JAX."""
    pytest.skip("glass.enumerate_spectra is not auto differentiable")

    n = 100
    tn = n * (n + 1) // 2

    # create mock spectra with 1 element counting to tn
    spectra: AngularPowerSpectra = [jnp.asarray(x) for x in range(tn)]

    def enumerate_spectra_no_generator(
        spectra: AngularPowerSpectra,
    ) -> list[tuple[int, int, AnyArray]]:
        return list(glass.enumerate_spectra(spectra))

    # iterator that will enumerate the spectra for checking
    jax.test_util.check_grads(
        enumerate_spectra_no_generator,
        (spectra,),
        order=1,
    )


def test_spectra_indices() -> None:
    """Tests glass.spectra_indices is auto differentiable when using JAX."""
    pytest.skip("glass.spectra_indices is not auto differentiable")

    jax.test_util.check_grads(
        partial(glass.spectra_indices, xp=jnp),
        (3,),
        order=1,
    )


@pytest.mark.parametrize("use_weights2", [True, False])
@pytest.mark.parametrize("lmax", [5, None])
def test_effective_cls(
    use_weights2: bool,
    lmax: int | None,
) -> None:
    """Tests glass.effective_cls is auto differentiable when using JAX."""
    cls: AngularPowerSpectra = [jnp.arange(15.0) for _ in range(3)]
    weights1 = jnp.ones((2, 1))
    weights2 = weights1 if use_weights2 else None

    jax.test_util.check_grads(
        partial(glass.effective_cls, lmax=lmax),
        (cls, weights1, weights2),
        order=1,
    )


def test_compute_gaussian_spectra() -> None:
    """Tests glass.compute_gaussian_spectra is auto differentiable when using JAX."""
    fields = [glass.grf.Normal(), glass.grf.Normal()]
    spectra: AngularPowerSpectra = [jnp.zeros(10) for _ in range(3)]

    jax.test_util.check_grads(
        partial(glass.compute_gaussian_spectra, fields),
        (spectra,),
        order=1,
    )


def test_solve_gaussian_spectra() -> None:
    """Tests glass.solve_gaussian_spectra is auto differentiable when using JAX."""
    pytest.skip("glass.solve_gaussian_spectra is not auto differentiable")

    fields = [glass.grf.Normal(), glass.grf.Normal()]
    spectra: AngularPowerSpectra = [jnp.zeros(5), jnp.zeros(10), jnp.zeros(15)]

    jax.test_util.check_grads(
        partial(glass.solve_gaussian_spectra, fields),
        (spectra,),
        order=1,
    )


@pytest.mark.parametrize("use_rng", [False, True])
@pytest.mark.parametrize("ncorr", [None, 1])
def test_generate(
    ncorr: int | None,
    use_rng: bool,
) -> None:
    """Tests glass.generate is auto differentiable when using JAX."""
    pytest.skip("glass.generate is not auto differentiable")

    gls: AngularPowerSpectra = [jnp.ones(10), jnp.ones(10), jnp.ones(10)]

    def generate_by_gls_no_generator(
        gls: AngularPowerSpectra,
    ) -> list[AnyArray]:
        fields = [lambda x, var: x, lambda x, var: x]  # noqa: ARG005
        # Must supply a fresh JAX RNG so that we generate the same sequence for each
        # call carried out by jax.test_util.check_grads
        rng = glass.rng.default_rng(xp=jnp) if use_rng else None
        return list(
            partial(glass.generate, fields, nside=16, ncorr=ncorr, rng=rng)(gls)
        )

    jax.test_util.check_grads(
        generate_by_gls_no_generator,
        (gls,),
        order=1,
    )


@pytest.mark.parametrize("lmax", [None, 1])
def test_cov_from_spectra(lmax: int | None) -> None:
    """Tests glass.cov_from_spectra is auto differentiable when using JAX."""
    pytest.skip("glass.cov_from_spectra is not auto differentiable")

    spectra: AngularPowerSpectra = [
        jnp.asarray(x)
        for x in [
            [110, 111, 112, 113],
            [220, 221, 222, 223],
            [210, 211, 212, 213],
            [330, 331, 332, 333],
            [320, 321, 322, 323],
            [310, 311, 312, 313],
        ]
    ]

    jax.test_util.check_grads(
        partial(glass.cov_from_spectra, lmax=lmax),
        (spectra,),
        order=1,
    )


@pytest.mark.parametrize("lmax", [None, 1])
# @pytest.mark.parametrize("method", ["nearest", "clip"])
def test_regularized_spectra_nearest(
    rng: UnifiedGenerator,
    lmax: int | None,
    # method: str,
) -> None:
    """Tests glass.regularized_spectra is auto differentiable when using JAX."""
    spectra: AngularPowerSpectra = [rng.random(20) for _ in range(6)]

    jax.test_util.check_grads(
        partial(glass.regularized_spectra, method="nearest", lmax=lmax, niter=300),
        (spectra,),
        order=1,
    )


@pytest.mark.parametrize("lmax", [None, 1])
def test_regularized_spectra_clip(
    rng: UnifiedGenerator,
    lmax: int | None,
    # method: str,
) -> None:
    """Tests glass.regularized_spectra is auto differentiable when using JAX."""
    spectra: AngularPowerSpectra = [rng.random(10) for _ in range(6)]

    jax.test_util.check_grads(
        partial(glass.regularized_spectra, method="clip", lmax=lmax),
        (spectra,),
        order=1,
    )
