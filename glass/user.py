# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''
User utilities (:mod:`glass.user`)
==================================

.. currentmodule:: glass.user

The :mod:`glass.user` module contains convenience functions for users of the
library.


Basic IO
----------------

.. autofunction:: save_cls
.. autofunction:: load_cls

FITS creation
----------------
.. autofunction:: awrite
.. autoclass:: AsyncHduWriter

'''

import numpy as np


def save_cls(filename, cls):
    '''Save a list of Cls to file.

    Uses :func:`numpy.savez` internally. The filename should therefore have a
    ``.npz`` suffix, or it will be given one.

    '''

    split = np.cumsum([len(cl) if cl is not None else 0 for cl in cls[:-1]])
    values = np.concatenate([cl for cl in cls if cl is not None])
    np.savez(filename, values=values, split=split)


def load_cls(filename):
    '''Load a list of Cls from file.

    Uses :func:`numpy.load` internally.

    '''

    with np.load(filename) as npz:
        values = npz['values']
        split = npz['split']
    return np.split(values, split)


"""Module for writing catalogue output."""

from contextlib import contextmanager
from threading import Thread

class AsyncHduWriter:
    """Writer that asynchronously appends rows to a HDU."""

    def __init__(self, fits, ext=None):
        """Create a new, uninitialised writer."""
        self.fits = fits
        self.ext = ext
        self.thread = None

    def _append(self, data, names=None):
        """Write routine for FITS data."""

        if self.ext is None or self.ext not in self.fits:
            self.fits.write_table(data, names=names, extname=self.ext)
            if self.ext is None:
                self.ext = self.fits[-1].get_extnum()
        else:
            hdu = self.fits[self.ext]
            # not using hdu.append here because of incompatibilities
            hdu.write(data, names=names, firstrow=hdu.get_nrows())

    def write(self, data=None, /, **columns):
        """Asynchronously append to FITS."""

        # if data is given, write it as it is
        if data is not None:
            if self.thread:
                self.thread.join()
            self.thread = Thread(target=self._append, args=(data,))
            self.thread.start()

        # if keyword arguments are given, treat them as names and columns
        if columns:
            names, values = list(columns.keys()), list(columns.values())
            if self.thread:
                self.thread.join()
            self.thread = Thread(target=self._append, args=(values, names))
            self.thread.start()


@contextmanager
def awrite(filename, *, ext=None):
    """Context manager for an asynchronous FITS catalogue writer."""

    import fitsio

    with fitsio.FITS(filename, "rw", clobber=True) as fits:
        fits.write(None)
        writer = AsyncHduWriter(fits, ext)
        try:
            yield writer
        finally:
            if writer.thread:
                writer.thread.join()
