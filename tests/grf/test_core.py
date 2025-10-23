from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

import glass.grf

if TYPE_CHECKING:
    import pytest_mock


def test_corr_unknown():
    class Unknown:
        def corr(self, _other, _x):  # type: ignore[no-untyped-def]
            return NotImplemented

        def icorr(self, _other, _x):  # type: ignore[no-untyped-def]
            return NotImplemented

        def dcorr(self, _other, _x):  # type: ignore[no-untyped-def]
            return NotImplemented

    t1 = glass.grf.Normal()
    t2 = Unknown()
    x = np.zeros(10)

    with pytest.raises(NotImplementedError, match="Unknown"):
        glass.grf.corr(t1, t2, x)

    with pytest.raises(NotImplementedError, match="Unknown"):
        glass.grf.icorr(t1, t2, x)

    with pytest.raises(NotImplementedError, match="Unknown"):
        glass.grf.dcorr(t1, t2, x)


def test_compute(mocker: pytest_mock.MockerFixture) -> None:
    cltocorr = mocker.patch("transformcl.cltocorr")
    icorr = mocker.patch("glass.grf._core.icorr")
    corrtocl = mocker.patch("transformcl.corrtocl")

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
