def test_getcl():
    from glass.fields import getcl
    # make a mock Cls array with the index pairs as entries
    cls = [{i, j} for i in range(10) for j in range(i, -1, -1)]
    # make sure indices are retrieved correctly
    for i in range(10):
        for j in range(10):
            assert getcl(cls, i, j) == {i, j}
