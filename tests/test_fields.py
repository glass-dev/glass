import numpy as np

from glass import getcl


def test_getcl() -> None:
    # make a mock Cls array with the index pairs as entries
    cls = [
        np.array([i, j], dtype=np.float64) for i in range(10) for j in range(i, -1, -1)
    ]
    # make sure indices are retrieved correctly
    for i in range(10):
        for j in range(10):
            result = getcl(cls, i, j)
            expected = np.array([min(i, j), max(i, j)], dtype=np.float64)
            np.testing.assert_array_equal(np.sort(result), expected)
