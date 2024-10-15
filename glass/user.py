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


def save_cls(filename: str, cls: list[npt.NDArray[np.float64] | None]) -> None:
    """
    Save a list of Cls to file.

    Uses :func:`numpy.savez` internally. The filename should therefore have a
    ``.npz`` suffix, or it will be given one.

    """
    split = np.cumsum([len(cl) if cl is not None else 0 for cl in cls[:-1]])
    values = np.concatenate([cl for cl in cls if cl is not None])
    np.savez(filename, values=values, split=split)


def load_cls(filename: str) -> list[npt.NDArray[np.float64]]:
    """
    Load a list of Cls from file.

    Uses :func:`numpy.load` internally.

    """
    with np.load(filename) as npz:
        values = npz["values"]
        split = npz["split"]
    return np.split(values, split)


class _FitsWriter:
    """
    Writer that creates a FITS file.

    Initialised with the fits object and extension name.
    """

    def __init__(self, fits, ext: str | None = None) -> None:  # type: ignore[no-untyped-def]
        """Create a new, uninitialised writer."""
        self.fits = fits
        self.ext = ext

    def _append(
        self,
        data: npt.NDArray[np.float64],
        names: list[str] | None = None,
    ) -> None:
        """Write the FITS file."""
        if self.ext is None or self.ext not in self.fits:
            self.fits.write_table(data, names=names, extname=self.ext)
            if self.ext is None:
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
        Write to FITS by calling the internal _append method.

        Pass either a positional variable (data)
        or multiple named arguments (**columns)
        """
        # if data is given, write it as it is
        if data is not None:
            self._append(data)

        # if keyword arguments are given, treat them as names and columns
        if columns:
            names, values = list(columns.keys()), list(columns.values())
            self._append(values, names)  # type: ignore[arg-type]


@contextmanager
def write_catalog(
    filename: str,
    *,
    ext: str | None = None,
) -> collections.abc.Generator[_FitsWriter]:
    """
    Write a catalogue into a FITS file.

    *ext* is the optional name of the extension.
    To be used as a context manager::

        # create the catalogue writer
        with write_catalog("catalog.fits") as out:
            ...
            # write catalogue columns RA, DEC, E1, E2, WHT with given arrays
            out.write(RA=lon, DEC=lat, E1=eps1, E2=e2, WHT=w)

    .. note::
       Requires the ``fitsio`` package.

    """
    import fitsio

    with fitsio.FITS(filename, "rw", clobber=True) as fits:
        fits.write(None)
        yield _FitsWriter(fits, ext)
