import numpy as np
import numpy.testing as npt


def test_tophat_windows():
    from glass.shells import tophat_windows

    zb = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0]
    dz = 0.005

    zp, wp = tophat_windows(zb, dz)

    assert len(zp) == len(wp) == len(zb)-1

    assert all(z0 == z[0] and zn == z[-1]
               for z, z0, zn in zip(zp, zb, zb[1:]))

    assert all(zn <= z0+len(z)*dz <= zn+dz
               for z, z0, zn in zip(zp, zb, zb[1:]))

    assert all(np.all(w == 1) for w in wp)


def test_restrict():
    from glass.shells import restrict

    # Gaussian test function
    z = np.linspace(0., 5., 1000)
    f = np.exp(-((z - 2.)/0.5)**2/2)

    # window for restriction
    za = [1., 2., 3., 4.]
    wa = [0., .5, .5, 0.]

    zr, fr = restrict(z, f, za, wa)

    assert zr[0] == za[0] and zr[-1] == za[-1]

    assert fr[0] == fr[-1] == 0.

    for zi, wi in zip(za, wa):
        i = np.searchsorted(zr, zi)
        assert zr[i] == zi
        assert fr[i] == wi*np.interp(zi, z, f)

    for zi, fi in zip(z, f):
        if za[0] <= zi <= za[-1]:
            i = np.searchsorted(zr, zi)
            assert zr[i] == zi
            assert fr[i] == fi*np.interp(zi, za, wa)


def test_partition():
    from glass.shells import partition

    # Gaussian test function
    z = np.linspace(0., 5., 1000)
    f = np.exp(-((z - 2.)/0.5)**2/2)

    # overlapping triangular weight functions
    zs = [[0., 1., 2.], [1., 2., 3.], [2., 3., 4.], [3., 4., 5.]]
    ws = [[0., 1., 0.], [0., 1., 0.], [0., 1., 0.], [0., 1., 0.]]

    zp, fp = partition(z, f, zs, ws)

    assert len(zp) == len(fp) == len(zs)

    for zr, za in zip(zp, zs):
        assert np.all((zr >= za[0]) & (zr <= za[-1]))

    for zr, fr, za, wa in zip(zp, fp, zs, ws):
        f_ = np.interp(zr, z, f, left=0., right=0.)
        w_ = np.interp(zr, za, wa, left=0., right=0.)
        npt.assert_allclose(fr, f_*w_)

    f_ = sum(np.interp(z, zr, fr, left=0., right=0.)
             for zr, fr in zip(zp, fp))

    # first and last points have zero total weight
    assert f_[0] == f_[-1] == 0.

    # find first and last index where total weight becomes unity
    i, j = np.searchsorted(z, [zs[0][1], zs[-1][1]])

    npt.assert_allclose(f_[i:j], f[i:j], atol=1e-15)
