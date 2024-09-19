import numpy as np
import pytest


def test_tophat_windows():
    from glass.shells import tophat_windows

    zb = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0]
    dz = 0.005

    ws = tophat_windows(zb, dz)

    assert len(ws) == len(zb) - 1

    assert all(z0 == w.za[0] and zn == w.za[-1] for w, z0, zn in zip(ws, zb, zb[1:]))

    assert all(
        zn <= z0 + len(w.za) * dz <= zn + dz for w, z0, zn in zip(ws, zb, zb[1:])
    )

    assert all(np.all(w.wa == 1) for w in ws)


def test_restrict():
    from glass.shells import RadialWindow, restrict

    # Gaussian test function
    z = np.linspace(0.0, 5.0, 1000)
    f = np.exp(-(((z - 2.0) / 0.5) ** 2) / 2)

    # window for restriction
    w = RadialWindow(za=[1.0, 2.0, 3.0, 4.0], wa=[0.0, 0.5, 0.5, 0.0], zeff=None)

    zr, fr = restrict(z, f, w)

    assert zr[0] == w.za[0]
    assert zr[-1] == w.za[-1]

    assert fr[0] == fr[-1] == 0.0

    for zi, wi in zip(w.za, w.wa):
        i = np.searchsorted(zr, zi)
        assert zr[i] == zi
        assert fr[i] == wi * np.interp(zi, z, f)

    for zi, fi in zip(z, f):
        if w.za[0] <= zi <= w.za[-1]:
            i = np.searchsorted(zr, zi)
            assert zr[i] == zi
            assert fr[i] == fi * np.interp(zi, w.za, w.wa)


@pytest.mark.parametrize("method", ["lstsq", "nnls", "restrict"])
def test_partition(method):
    import numpy as np

    from glass.shells import RadialWindow, partition

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

    assert np.allclose(part.sum(axis=0), np.trapz(fz, z))
