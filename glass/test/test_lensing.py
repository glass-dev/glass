import numpy as np
import numpy.testing as npt
import pytest


@pytest.mark.parametrize("usecomplex", [True, False])
def test_deflect_nsew(usecomplex):
    from glass.lensing import deflect

    d = 5.
    r = np.radians(d)

    if usecomplex:
        def alpha(re, im):
            return re + 1j*im
    else:
        def alpha(re, im):
            return [re, im]

    # north
    lon, lat = deflect(0., 0., alpha(r, 0))
    assert np.allclose([lon, lat], [0., d])

    # south
    lon, lat = deflect(0., 0., alpha(-r, 0))
    assert np.allclose([lon, lat], [0., -d])

    # east
    lon, lat = deflect(0., 0., alpha(0, r))
    assert np.allclose([lon, lat], [-d, 0.])

    # west
    lon, lat = deflect(0., 0., alpha(0, -r))
    assert np.allclose([lon, lat], [d, 0.])


def test_deflect_many():
    import healpix
    from glass.lensing import deflect

    n = 1000
    abs_alpha = np.random.uniform(0, 2*np.pi, size=n)
    arg_alpha = np.random.uniform(-np.pi, np.pi, size=n)

    lon_ = np.degrees(np.random.uniform(-np.pi, np.pi, size=n))
    lat_ = np.degrees(np.arcsin(np.random.uniform(-1, 1, size=n)))

    lon, lat = deflect(lon_, lat_, abs_alpha*np.exp(1j*arg_alpha))

    x_, y_, z_ = healpix.ang2vec(lon_, lat_, lonlat=True)
    x, y, z = healpix.ang2vec(lon, lat, lonlat=True)

    dotp = x*x_ + y*y_ + z*z_

    npt.assert_allclose(dotp, np.cos(abs_alpha))
