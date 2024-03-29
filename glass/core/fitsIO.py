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
