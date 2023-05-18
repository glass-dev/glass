def test_positions_from_delta():
    import numpy as np
    from glass.points import positions_from_delta

    # case 1: test single-dimensional input

    ngal = 1e-3
    delta = np.zeros(12)
    bias = 0.8
    vis = np.ones(12)

    lon, lat, cnt = positions_from_delta(ngal, delta, bias, vis)

    assert isinstance(cnt, np.integer)
    assert lon.shape == lat.shape == (cnt,)

    # case 2: test multi-dimensional ngal

    ngal = [1e-3, 2e-3]
    delta = np.zeros(12)
    bias = 0.8
    vis = np.ones(12)

    lon, lat, cnt = positions_from_delta(ngal, delta, bias, vis)

    assert isinstance(cnt, list)
    assert isinstance(lon, list)
    assert isinstance(lat, list)
    assert len(cnt) == len(lon) == len(lat) == 2
    for i, c in enumerate(cnt):
        assert isinstance(c, np.integer)
        assert lon[i].shape == lat[i].shape == (c,)

    # case 3: test multi-dimensional delta

    ngal = 1e-3
    delta = np.zeros((2, 12))
    bias = 0.8
    vis = np.ones(12)

    lon, lat, cnt = positions_from_delta(ngal, delta, bias, vis)

    assert isinstance(cnt, list)
    assert isinstance(lon, list)
    assert isinstance(lat, list)
    assert len(cnt) == len(lon) == len(lat) == 2
    for i, c in enumerate(cnt):
        assert isinstance(c, np.integer)
        assert lon[i].shape == lat[i].shape == (c,)

    # case 4: test multi-dimensional broadcasting

    ngal = [1e-3, 2e-3]
    delta = np.zeros((3, 1, 12))
    bias = 0.8
    vis = np.ones(12)

    lon, lat, cnt = positions_from_delta(ngal, delta, bias, vis)

    assert isinstance(cnt, list)
    assert isinstance(lon, list)
    assert isinstance(lat, list)
    assert len(cnt) == len(lon) == len(lat) == 6
    for i, c in enumerate(cnt):
        assert isinstance(c, np.integer)
        assert lon[i].shape == lat[i].shape == (c,)


def test_uniform_positions():
    import numpy as np
    from glass.points import uniform_positions

    # case 1: test scalar input

    ngal = 1e-3

    lon, lat, cnt = uniform_positions(ngal)

    assert isinstance(cnt, np.integer)
    assert lon.shape == lat.shape == (cnt,)

    # case 2: test 1-D array input

    ngal = [1e-3, 2e-3, 3e-3]

    lon, lat, cnt = uniform_positions(ngal)

    assert isinstance(cnt, list)
    assert isinstance(lon, list)
    assert isinstance(lat, list)
    assert len(cnt) == len(lon) == len(lat) == 3
    for i, c in enumerate(cnt):
        assert isinstance(c, np.integer)
        assert lon[i].shape == lat[i].shape == (c,)

    # case 3: test 2-D array input

    ngal = [[1e-3, 2e-3], [3e-3, 4e-3]]

    lon, lat, cnt = uniform_positions(ngal)

    assert isinstance(cnt, list)
    assert isinstance(lon, list)
    assert isinstance(lat, list)
    assert len(cnt) == len(lon) == len(lat) == 4
    for i, c in enumerate(cnt):
        assert isinstance(c, np.integer)
        assert lon[i].shape == lat[i].shape == (c,)
