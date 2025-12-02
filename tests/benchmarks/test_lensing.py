from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import glass.lensing

if TYPE_CHECKING:
    from types import ModuleType
    from typing import Never

    from conftest import Compare
    from pytest_benchmark.fixture import BenchmarkFixture

    from cosmology import Cosmology

    from glass._types import FloatArray, UnifiedGenerator


@pytest.mark.stable
def test_multi_plane_matrix(
    benchmark: BenchmarkFixture,
    compare: type[Compare],
    cosmo: Cosmology,
    urng: UnifiedGenerator,
    xp: ModuleType,
) -> None:
    """Benchmarks for add_window and add_plane with a multi_plane_matrix."""
    if xp.__name__ == "array_api_strict":
        pytest.skip(
            f"glass.lensing.multi_plane_matrix not yet ported for {xp.__name__}"
        )

    # Use this over the fixture to allow us to add many more windows
    shells = [
        glass.RadialWindow(
            xp.arange(i, i + 3, dtype=xp.float64),
            xp.asarray([0.0, 1.0, 0.0]),
            float(i + 1),
        )
        for i in range(1000)
    ]
    mat = glass.multi_plane_matrix(shells, cosmo)
    deltas = urng.random((len(shells), 10))

    compare.assert_array_equal(mat, xp.tril(mat))
    compare.assert_array_equal(xp.triu(mat, 1), 0)

    def setup_shells_and_deltas() -> tuple[
        tuple[
            glass.lensing.MultiPlaneConvergence,
            zip[tuple[glass.RadialWindow, FloatArray]],
        ],
        dict[Never, Never],
    ]:
        """Run setup a generator with zip before each benchmark run."""
        convergence = glass.lensing.MultiPlaneConvergence(cosmo)
        return (convergence, zip(shells, deltas, strict=False)), {}

    def multi_plane_matrix_add_window(
        convergence: type[glass.lensing.MultiPlaneConvergence],
        zipped: tuple[list[type[glass.RadialWindow]], FloatArray],
    ) -> type[glass.lensing.MultiPlaneConvergence]:
        """Call add_window repeatedly, to be benchmarked."""
        for shell, delta in zipped:
            convergence.add_window(delta, shell)  # type: ignore[call-arg,arg-type]
        return convergence

    actual_convergence = benchmark.pedantic(
        multi_plane_matrix_add_window,
        setup=setup_shells_and_deltas,
        rounds=500,
    )

    assert len(actual_convergence.kappa) == 10
    for x in actual_convergence.kappa:
        assert x is not None


@pytest.mark.stable
def test_multi_plane_weights(
    benchmark: BenchmarkFixture,
    compare: type[Compare],
    cosmo: Cosmology,
    urng: UnifiedGenerator,
    xp: ModuleType,
) -> None:
    """Benchmarks for add_window and add_plane with a multi_plane_weights."""
    if xp.__name__ in {"array_api_strict", "jax.numpy"}:
        pytest.skip(
            f"glass.lensing.multi_plane_weights not yet ported for {xp.__name__}"
        )

    # Use this over the fixture to allow us to add many more windows
    shells = [
        glass.RadialWindow(
            xp.arange(i, i + 3, dtype=xp.float64),
            xp.asarray([0.0, 1.0, 0.0]),
            float(i + 1),
        )
        for i in range(500)
    ]
    w_in = xp.eye(len(shells))
    deltas = urng.random((len(shells), 10))
    weights = urng.random((len(shells), 3))

    w_out = glass.multi_plane_weights(w_in, shells, cosmo)

    compare.assert_array_equal(w_out, xp.triu(w_out, 1))
    compare.assert_array_equal(xp.tril(w_out), 0)

    def setup_shells_deltas_and_weights() -> tuple[
        tuple[
            glass.lensing.MultiPlaneConvergence,
            zip[tuple[glass.RadialWindow, FloatArray, FloatArray]],
        ],
        dict[Never, Never],
    ]:
        """Run setup a generator with zip before each benchmark run."""
        convergence = glass.lensing.MultiPlaneConvergence(cosmo)
        return (convergence, zip(shells, deltas, weights, strict=False)), {}

    def multi_plane_weights_add_window(
        convergence: type[glass.lensing.MultiPlaneConvergence],
        zipped: tuple[list[type[glass.RadialWindow]], FloatArray, FloatArray],
    ) -> type[glass.lensing.MultiPlaneConvergence]:
        """Call add_window repeatedly, to be benchmarked."""
        for shell, delta, _ in zipped:
            convergence.add_window(delta, shell)  # type: ignore[call-arg,arg-type]
        return convergence

    actual_convergence = benchmark.pedantic(
        multi_plane_weights_add_window,
        setup=setup_shells_deltas_and_weights,
        rounds=100,
    )

    assert len(actual_convergence.kappa) == 10
    for x in actual_convergence.kappa:
        assert x is not None
