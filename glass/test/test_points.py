def test_positions_from_delta():
    import numpy as np
    from glass.points import positions_from_delta

    # case: single-dimensional input

    ngal = 1e-3
    delta = np.zeros(12)
    bias = 0.8
    vis = np.ones(12)

    lon, lat, cnt = positions_from_delta(ngal, delta, bias, vis)

    assert isinstance(cnt, int)
    assert lon.shape == lat.shape == (cnt,)

    # case: multi-dimensional ngal

    ngal = [1e-3, 2e-3]
    delta = np.zeros(12)
    bias = 0.8
    vis = np.ones(12)

    lon, lat, cnt = positions_from_delta(ngal, delta, bias, vis)

    assert cnt.shape == (2,)
    assert lon.shape == lat.shape == (cnt.sum(),)

    # case: multi-dimensional delta

    ngal = 1e-3
    delta = np.zeros((3, 2, 12))
    bias = 0.8
    vis = np.ones(12)

    lon, lat, cnt = positions_from_delta(ngal, delta, bias, vis)

    assert cnt.shape == (3, 2)
    assert lon.shape == lat.shape == (cnt.sum(),)

    # case: multi-dimensional broadcasting

    ngal = [1e-3, 2e-3]
    delta = np.zeros((3, 1, 12))
    bias = 0.8
    vis = np.ones(12)

    lon, lat, cnt = positions_from_delta(ngal, delta, bias, vis)

    assert cnt.shape == (3, 2)
    assert lon.shape == lat.shape == (cnt.sum(),)


def test_uniform_positions():
    from glass.points import uniform_positions

    # case: scalar input

    ngal = 1e-3

    lon, lat, cnt = uniform_positions(ngal)

    assert isinstance(cnt, int)
    assert lon.shape == lat.shape == (cnt,)

    # case: 1-D array input

    ngal = [1e-3, 2e-3, 3e-3]

    lon, lat, cnt = uniform_positions(ngal)

    assert cnt.shape == (3,)
    assert lon.shape == lat.shape == (cnt.sum(),)

    # case: 2-D array input

    ngal = [[1e-3, 2e-3], [3e-3, 4e-3], [5e-3, 6e-3]]

    lon, lat, cnt = uniform_positions(ngal)

    assert cnt.shape == (3, 2)
    assert lon.shape == lat.shape == (cnt.sum(),)
