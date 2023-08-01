import numpy as np


def test_tophat_windows():
    from glass.shells import tophat_windows

    zb = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0]
    dz = 0.005

    ws = tophat_windows(zb, dz)

    assert len(ws) == len(zb)-1

    assert all(z0 == w.za[0] and zn == w.za[-1]
               for w, z0, zn in zip(ws, zb, zb[1:]))

    assert all(zn <= z0+len(w.za)*dz <= zn+dz
               for w, z0, zn in zip(ws, zb, zb[1:]))

    assert all(np.all(w.wa == 1) for w in ws)


def test_restrict():
    from glass.shells import restrict, RadialWindow

    # Gaussian test function
    z = np.linspace(0., 5., 1000)
    f = np.exp(-((z - 2.)/0.5)**2/2)

    # window for restriction
    w = RadialWindow(za=[1., 2., 3., 4.], wa=[0., .5, .5, 0.], zeff=None)

    zr, fr = restrict(z, f, w)

    assert zr[0] == w.za[0] and zr[-1] == w.za[-1]

    assert fr[0] == fr[-1] == 0.

    for zi, wi in zip(w.za, w.wa):
        i = np.searchsorted(zr, zi)
        assert zr[i] == zi
        assert fr[i] == wi*np.interp(zi, z, f)

    for zi, fi in zip(z, f):
        if w.za[0] <= zi <= w.za[-1]:
            i = np.searchsorted(zr, zi)
            assert zr[i] == zi
            assert fr[i] == fi*np.interp(zi, w.za, w.wa)
