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


def test_multi_plane_matrix_add_window(
    benchmark: BenchmarkFixture,
    compare: type[Compare],
    cosmo: Cosmology,
    urng: UnifiedGenerator,
    xp: ModuleType,
) -> None:
    """Benchmarks for MultiPlaneConvergence.add_window with a multi_plane_matrix."""
    if xp.__name__ == "array_api_strict":
        pytest.skip(f"glass.fields.generate not yet ported for {xp.__name__}")

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

    def setup_shells_and_delta() -> tuple[
        tuple[
            glass.lensing.MultiPlaneConvergence,
            zip[tuple[glass.RadialWindow, FloatArray]],
        ],
        dict[Never, Never],
    ]:
        """Run setup a generator with zip before each benchmark run."""
        convergence = glass.lensing.MultiPlaneConvergence(cosmo)
        return (convergence, zip(shells, deltas, strict=False)), {}

    def function_to_benchmark(
        convergence: type[glass.lensing.MultiPlaneConvergence],
        zipped: tuple[list[type[glass.RadialWindow]], FloatArray],
    ) -> type[glass.lensing.MultiPlaneConvergence]:
        """Call add_window repeatedly, to be benchmarked."""
        for shell, delta in zipped:
            convergence.add_window(delta, shell)  # type: ignore[call-arg,arg-type]
        return convergence

    actual_convergence = benchmark.pedantic(
        function_to_benchmark,
        setup=setup_shells_and_delta,
        rounds=500,
    )

    # This was generated on the first run of this test rather than calculated
    expected_kappa = [
        3427039.750252,
        3431185.04507,
        3492292.408525,
        3415536.190037,
        3421145.593983,
        3411650.553902,
        3432103.932316,
        3343221.577821,
        3441975.126385,
        3462143.319128,
    ]
    compare.assert_allclose(actual_convergence.kappa, expected_kappa)
