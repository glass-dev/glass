import numpy as np
import numpy.testing as npt


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


def test_partition():
    from glass.shells import partition, RadialWindow

    # Gaussian test function
    z = np.linspace(0., 5., 1000)
    f = np.exp(-((z - 2.)/0.5)**2/2)

    # overlapping triangular weight functions
    ws = [RadialWindow(za=[0., 1., 2.], wa=[0., 1., 0.], zeff=None),
          RadialWindow(za=[1., 2., 3.], wa=[0., 1., 0.], zeff=None),
          RadialWindow(za=[2., 3., 4.], wa=[0., 1., 0.], zeff=None),
          RadialWindow(za=[3., 4., 5.], wa=[0., 1., 0.], zeff=None)]

    zp, fp = partition(z, f, ws)

    assert len(zp) == len(fp) == len(ws)

    for zr, w in zip(zp, ws):
        assert np.all((zr >= w.za[0]) & (zr <= w.za[-1]))

    for zr, fr, w in zip(zp, fp, ws):
        f_ = np.interp(zr, z, f, left=0., right=0.)
        w_ = np.interp(zr, w.za, w.wa, left=0., right=0.)
        npt.assert_allclose(fr, f_*w_)

    f_ = sum(np.interp(z, zr, fr, left=0., right=0.)
             for zr, fr in zip(zp, fp))

    # first and last points have zero total weight
    assert f_[0] == f_[-1] == 0.

    # find first and last index where total weight becomes unity
    i, j = np.searchsorted(z, [ws[0].za[1], ws[-1].za[1]])

    npt.assert_allclose(f_[i:j], f[i:j], atol=1e-15)
