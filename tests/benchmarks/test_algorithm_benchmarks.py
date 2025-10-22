from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

import glass.algorithm

if TYPE_CHECKING:
    import types


@pytest.mark.benchmark(
    group="algorithm.nnls",
)
def test_nnls(xp: types.ModuleType, benchmark) -> None:
    if xp.__name__ == "jax.numpy":
        pytest.skip("Arrays in nnls are not immutable, so do not support jax")

    # a is 0.0->50.0 with a negative number instead of multiples of 7
    a = xp.arange(0.0, 500.0, 1.0)
    a = xp.where((xp.astype(a, xp.int64) % 7) == 0, -1.1781, a)
    a = xp.reshape(a, (50, 10))
    # b is 0.0->10.0 with zero instead of multiples of 2
    b = xp.arange(0.0, 10.0, 1.0)
    b = xp.where((xp.astype(b, xp.int64) % 2) == 0, b, 0.0)
    x = a @ b
    res = benchmark(
        glass.algorithm.nnls,
        a,
        x,
        tol=500 * xp.linalg.matrix_norm(a, ord=1) * xp.finfo(xp.float64).eps,
    )
    np.testing.assert_allclose(
        res,
        xp.asarray(
            [0.0, 0.0, 0.0, 0.0, 4.002368, 0.0, 5.998561, 0.0, 8.000571, 1.957038]
        ),
        rtol=1e-6,
    )
