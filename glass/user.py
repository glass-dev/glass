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

"""  # noqa: D400

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import importlib.util
    import pathlib
    from collections.abc import Generator, Sequence

    from numpy.typing import NDArray

    if importlib.util.find_spec("fitsio") is not None:
        import fitsio


def save_cls(
    filename: str,
    cls: Sequence[NDArray[np.float64] | Sequence[float]],
) -> None:
    """
    Save a list of Cls to file.

    Uses :func:`numpy.savez` internally. The filename should therefore have a
    ``.npz`` suffix, or it will be given one.

    Parameters
    ----------
    filename
        The name of the file to save to.
    cls
        Angular matter power spectra in *GLASS* ordering.

    """
    split = np.cumsum([len(cl) for cl in cls[:-1]])
    values = np.concatenate(cls)
    np.savez(filename, values=values, split=split)


def load_cls(
    filename: str,
) -> list[NDArray[np.float64] | Sequence[float]]:
    """
    Load a list of Cls from file.

    Uses :func:`numpy.load` internally.

    Parameters
    ----------
    filename
        The name of the file to load from.

    Returns
    -------
        The list of Cls.

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

    def __init__(self, fits: fitsio.FITS, ext: str | None = None) -> None:
        """
        Create a new, uninitialised writer.

        Parameters
        ----------
        fits
            The fits object.
        ext
            The file extension.

        """
        self.fits = fits
        self.ext = ext

    def _append(
        self,
        data: NDArray[np.float64] | list[NDArray[np.float64]],
        names: list[str] | None = None,
    ) -> None:
        """
        Write the FITS file.

        Parameters
        ----------
        data
            The data to write.
        names
            The names of the columns.

        """
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
        data: NDArray[np.float64] | None = None,
        /,
        **columns: NDArray[np.float64],
    ) -> None:
        """
        Write to FITS by calling the internal _append method.

        Pass either a positional variable (data)
        or multiple named arguments (**columns)

        Parameters
        ----------
        data
            The data to write.
        columns
            The columns to write.

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
    ext: str | None = None,
) -> Generator[_FitsWriter]:
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

    Parameters
    ----------
    filename
        The name of the file to write to.
    ext
        The file extension.

    Yields
    ------
    writer
        The writer object.

    """
    import fitsio  # noqa: PLC0415

    with fitsio.FITS(filename, "rw", clobber=True) as fits:
        fits.write(None)
        yield _FitsWriter(fits, ext)
