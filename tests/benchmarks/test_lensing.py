from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import glass

if TYPE_CHECKING:
    from types import ModuleType

    from pytest_benchmark.fixture import BenchmarkFixture
    from typing_extensions import Never

    from glass._types import FloatArray, UnifiedGenerator
    from glass.cosmology import Cosmology
    from tests.fixtures.helper_classes import Compare


@pytest.mark.stable
def test_multi_plane_matrix(
    benchmark: BenchmarkFixture,
    compare: type[Compare],
    cosmo: Cosmology,
    urngb: UnifiedGenerator,
    xpb: ModuleType,
) -> None:
    """Benchmarks for add_window and add_plane with a multi_plane_matrix."""
    # Use this over the fixture to allow us to add many more windows
    shells = [
        glass.RadialWindow(
            xpb.arange(i, i + 3, dtype=xpb.float64),
            xpb.asarray([0.0, 1.0, 0.0]),
            float(i + 1),
        )
        for i in range(1_000)
    ]
    mat = glass.multi_plane_matrix(shells, cosmo)
    deltas = urngb.random((len(shells), 10))

    compare.assert_array_equal(mat, xpb.tril(mat))
    compare.assert_array_equal(xpb.triu(mat, k=1), 0)

    def setup_shells_and_deltas() -> tuple[
        tuple[
            glass.MultiPlaneConvergence,
            list[glass.RadialWindow],
            FloatArray,
        ],
        dict[Never, Never],
    ]:
        """Run setup a generator with zip before each benchmark run."""
        convergence = glass.MultiPlaneConvergence(cosmo)
        return (convergence, shells, deltas), {}

    def multi_plane_matrix_add_window(
        convergence: type[glass.MultiPlaneConvergence],
        shells: list[type[glass.RadialWindow]],
        deltas: FloatArray,
    ) -> type[glass.MultiPlaneConvergence]:
        """Call add_window repeatedly, to be benchmarked."""
        for i in range(min(len(shells), deltas.shape[0])):
            shell = shells[i]
            delta = deltas[i, ...]
            convergence.add_window(delta, shell)  # type: ignore[arg-type,call-arg]
        return convergence

    actual_convergence = benchmark.pedantic(
        multi_plane_matrix_add_window,
        setup=setup_shells_and_deltas,
        rounds=500,
    )

    assert actual_convergence.kappa.shape[0] == 10
    for x in actual_convergence.kappa:
        assert x is not None


@pytest.mark.stable
def test_multi_plane_weights(
    benchmark: BenchmarkFixture,
    compare: type[Compare],
    cosmo: Cosmology,
    urngb: UnifiedGenerator,
    xpb: ModuleType,
) -> None:
    """Benchmarks for add_window and add_plane with a multi_plane_weights."""
    # Use this over the fixture to allow us to add many more windows
    shells = [
        glass.RadialWindow(
            xpb.arange(i, i + 3, dtype=xpb.float64),
            xpb.asarray([0.0, 1.0, 0.0]),
            float(i + 1),
        )
        for i in range(500)
    ]
    w_in = xpb.eye(len(shells))
    deltas = urngb.random((len(shells), 10))
    weights = urngb.random((len(shells), 3))

    w_out = glass.multi_plane_weights(w_in, shells, cosmo)

    compare.assert_array_equal(w_out, xpb.triu(w_out, 1))
    compare.assert_array_equal(xpb.tril(w_out), 0)

    def setup_shells_deltas_and_weights() -> tuple[
        tuple[
            glass.MultiPlaneConvergence,
            zip[tuple[glass.RadialWindow, FloatArray, FloatArray]],
        ],
        dict[Never, Never],
    ]:
        """Run setup a generator with zip before each benchmark run."""
        convergence = glass.MultiPlaneConvergence(cosmo)
        return (convergence, zip(shells, deltas, weights, strict=False)), {}

    def multi_plane_weights_add_window(
        convergence: type[glass.MultiPlaneConvergence],
        zipped: tuple[list[type[glass.RadialWindow]], FloatArray, FloatArray],
    ) -> type[glass.MultiPlaneConvergence]:
        """Call add_window repeatedly, to be benchmarked."""
        for shell, delta, _ in zipped:
            convergence.add_window(delta, shell)  # type: ignore[arg-type,call-arg]
        return convergence

    actual_convergence = benchmark.pedantic(
        multi_plane_weights_add_window,
        setup=setup_shells_deltas_and_weights,
        rounds=100,
    )

    assert len(actual_convergence.kappa) == 10
    for x in actual_convergence.kappa:
        assert x is not None
