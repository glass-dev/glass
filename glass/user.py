"""
User utilities
==============

.. currentmodule:: glass

The following functions/classes provide convenience functionality for users
of the library.


Input and Output
----------------

.. autofunction:: save_cls
.. autofunction:: load_cls
.. autofunction:: write_catalog

"""  # noqa: D205, D400

from __future__ import annotations

import typing
from contextlib import contextmanager

import numpy as np
import numpy.typing as npt

if typing.TYPE_CHECKING:
    import collections.abc
    import importlib.util
    import pathlib

    if importlib.util.find_spec("fitsio") is not None:
        import fitsio


def save_cls(filename: str, cls: list[npt.NDArray[np.float64] | None]) -> None:
    """
    _summary_.

    Parameters
    ----------
    filename
        _description_
    cls
        _description_

    """
    split = np.cumsum([len(cl) if cl is not None else 0 for cl in cls[:-1]])
    values = np.concatenate([cl for cl in cls if cl is not None])
    np.savez(filename, values=values, split=split)


def load_cls(filename: str) -> list[npt.NDArray[np.float64]]:
    """
    _summary_.

    Parameters
    ----------
    filename
        _description_

    Returns
    -------
        _description_

    """
    with np.load(filename) as npz:
        values = npz["values"]
        split = npz["split"]
    return np.split(values, split)


class _FitsWriter:
    """_summary_."""

    def __init__(self, fits: fitsio.FITS, ext: str = "") -> None:
        """
        _summary_.

        Parameters
        ----------
        fits
            _description_
        ext
            _description_

        """
        self.fits = fits
        self.ext = ext

    def _append(
        self,
        data: npt.NDArray[np.float64] | list[npt.NDArray[np.float64]],
        names: list[str] | None = None,
    ) -> None:
        """
        _summary_.

        Parameters
        ----------
        data
            _description_
        names
            _description_

        """
        if self.ext not in self.fits:
            self.fits.write_table(data, names=names, extname=self.ext)
            if not self.ext:
                self.ext = self.fits[-1].get_extnum()
        else:
            hdu = self.fits[self.ext]
            # not using hdu.append here because of incompatibilities
            hdu.write(data, names=names, firstrow=hdu.get_nrows())

    def write(
        self,
        data: npt.NDArray[np.float64] | None = None,
        /,
        **columns: npt.NDArray[np.float64],
    ) -> None:
        """
        _summary_.

        Parameters
        ----------
        data
            _description_

        """
        # if data is given, write it as it is
        if data is not None:
            self._append(data)

        # if keyword arguments are given, treat them as names and columns
        if columns:
            names, values = list(columns.keys()), list(columns.values())
            self._append(values, names)


@contextmanager
def write_catalog(
    filename: pathlib.Path,
    *,
    ext: str = "",
) -> collections.abc.Generator[_FitsWriter]:
    """
    _summary_.

    Parameters
    ----------
    filename
        _description_
    ext
        _description_

    Returns
    -------
        _description_

    Yields
    ------
        _description_

    """
    import fitsio

    with fitsio.FITS(filename, "rw", clobber=True) as fits:
        fits.write(None)
        yield _FitsWriter(fits, ext)
