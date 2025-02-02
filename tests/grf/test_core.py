import unittest.mock

import numpy as np
import pytest

import glass.grf


def test_dispatch():
    @glass.grf._core.dispatch
    def test(_a, _b, _c):
        return None

    @test.add
    def _(_a: int, _b: str, _c: object):
        return 1

    @test.add
    def _(_a: list, _b: tuple, _c: object):
        return 2

    # first match
    assert test(1, "1", ...) == 1
    # first match inverted
    assert test("1", 1, ...) == 1
    # second match
    assert test([], (), ...) == 2
    # second match inverted
    assert test((), [], ...) == 2
    # mismatch
    assert test(..., ..., ...) is None
    # partial mismatch
    assert test(1, 2, ...) is None
    # cross mismatch
    assert test(1, (), ...) is None


def test_dispatch_bad_function():
    test = glass.grf._core.dispatch(unittest.mock.Mock())

    with pytest.raises(TypeError):

        @test.add
        def bad_signature(_a, _b): ...

    with pytest.raises(TypeError):

        @test.add
        def bad_annotation(_a: int, _b, _c: str): ...


def test_corr_unknown():
    class Unknown:
        pass

    t1 = glass.grf.Normal()
    t2 = Unknown()
    x = np.zeros(10)

    with pytest.raises(NotImplementedError, match="Unknown"):
        glass.grf.corr(t1, t2, x)

    with pytest.raises(NotImplementedError, match="Unknown"):
        glass.grf.icorr(t1, t2, x)

    with pytest.raises(NotImplementedError, match="Unknown"):
        glass.grf.dcorr(t1, t2, x)


@unittest.mock.patch("transformcl.corrtocl")
@unittest.mock.patch("glass.grf._core.icorr")
@unittest.mock.patch("transformcl.cltocorr")
def test_compute(cltocorr, icorr, corrtocl):
    t1 = glass.grf.Normal()
    t2 = glass.grf.Normal()
    x = np.zeros(10)

    result = glass.grf.compute(x, t1, t2)

    cltocorr.assert_called_once_with(x)
    icorr.assert_called_once_with(t1, t2, cltocorr.return_value)
    corrtocl.assert_called_once_with(icorr.return_value)
    assert result is corrtocl.return_value

    cltocorr.reset_mock()
    icorr.reset_mock()
    corrtocl.reset_mock()

    # default t2
    result = glass.grf.compute(x, t1)

    cltocorr.assert_called_once_with(x)
    icorr.assert_called_once_with(t1, t1, cltocorr.return_value)
    corrtocl.assert_called_once_with(icorr.return_value)
    assert result is corrtocl.return_value
