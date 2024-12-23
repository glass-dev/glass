import numpy as np
import pytest

from glass import (
    RadialWindow,
    partition,
    restrict,
    tophat_windows,
)


def test_tophat_windows() -> None:
    zb = np.array([0.0, 0.1, 0.2, 0.5, 1.0, 2.0])
    dz = 0.005

    ws = tophat_windows(zb, dz)

    assert len(ws) == len(zb) - 1

    assert all(
        z0 == w.za[0] and zn == w.za[-1]
        for w, z0, zn in zip(ws, zb, zb[1:], strict=False)
    )

    assert all(
        zn <= z0 + len(w.za) * dz <= zn + dz
        for w, z0, zn in zip(ws, zb, zb[1:], strict=False)
    )

    assert all(np.all(w.wa == 1) for w in ws)


def test_restrict() -> None:
    # Gaussian test function
    z = np.linspace(0.0, 5.0, 1000)
    f = np.exp(-(((z - 2.0) / 0.5) ** 2) / 2)

    # window for restriction
    w = RadialWindow(
        za=np.array([1.0, 2.0, 3.0, 4.0]),
        wa=np.array([0.0, 0.5, 0.5, 0.0]),
    )

    zr, fr = restrict(z, f, w)

    assert zr[0] == w.za[0]
    assert zr[-1] == w.za[-1]

    assert fr[0] == fr[-1] == 0.0

    for zi, wi in zip(w.za, w.wa, strict=False):
        i = np.searchsorted(zr, zi)
        assert zr[i] == zi
        assert fr[i] == wi * np.interp(zi, z, f)

    for zi, fi in zip(z, f, strict=False):
        if w.za[0] <= zi <= w.za[-1]:
            i = np.searchsorted(zr, zi)
            assert zr[i] == zi
            assert fr[i] == fi * np.interp(zi, w.za, w.wa)


@pytest.mark.parametrize("method", ["lstsq", "nnls", "restrict"])
def test_partition(method: str) -> None:
    shells = [
        RadialWindow(np.array([0.0, 1.0]), np.array([1.0, 0.0]), 0.0),
        RadialWindow(np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 0.0]), 0.5),
        RadialWindow(np.array([1.0, 2.0, 3.0]), np.array([0.0, 1.0, 0.0]), 1.5),
        RadialWindow(np.array([2.0, 3.0, 4.0]), np.array([0.0, 1.0, 0.0]), 2.5),
        RadialWindow(np.array([3.0, 4.0, 5.0]), np.array([0.0, 1.0, 0.0]), 3.5),
        RadialWindow(np.array([4.0, 5.0]), np.array([0.0, 1.0]), 5.0),
    ]

    z = np.linspace(0.0, 5.0, 1000)
    k = 1 + np.arange(6).reshape(3, 2, 1)
    fz = np.exp(-z / k)

    assert fz.shape == (3, 2, 1000)

    part = partition(z, fz, shells, method=method)

    assert part.shape == (len(shells), 3, 2)

    np.testing.assert_allclose(part.sum(axis=0), np.trapezoid(fz, z))
