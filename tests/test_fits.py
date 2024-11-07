import importlib.util
import pathlib

import numpy as np
import numpy.typing as npt
import pytest

from glass import user

# check if fitsio is available for testing
HAVE_FITSIO = importlib.util.find_spec("fitsio") is not None

delta = 0.001  # Number of points in arrays
my_max = 1000  # Typically number of galaxies in loop
except_int = 750  # Where test exception occurs in loop
filename = "MyFile.Fits"


@pytest.mark.skipif(not HAVE_FITSIO, reason="test requires fitsio")
def test_basic_write(tmp_path: pathlib.Path) -> None:
    import fitsio

    filename_gfits = "gfits.fits"  # what GLASS creates
    filename_tfits = "tfits.fits"  # file created on the fly to test against

    def _test_append(
        fits: fitsio.FITS,
        data: list[npt.NDArray[np.float64]],
        names: list[str],
    ) -> None:
        """Write routine for FITS test cases."""
        cat_name = "CATALOG"
        if cat_name not in fits:
            fits.write_table(data, names=names, extname=cat_name)
        else:
            hdu = fits[cat_name]
            hdu.write(data, names=names, firstrow=hdu.get_nrows())

    with (
        user.write_catalog(tmp_path / filename_gfits, ext="CATALOG") as out,
        fitsio.FITS(tmp_path / filename_tfits, "rw", clobber=True) as my_fits,
    ):
        for i in range(my_max):
            array = np.arange(i, i + 1, delta)  # array of size 1/delta
            array2 = np.arange(i + 1, i + 2, delta)  # array of size 1/delta
            out.write(RA=array, RB=array2)
            arrays = [array, array2]
            names = ["RA", "RB"]
            _test_append(my_fits, arrays, names)

    with (
        fitsio.FITS(tmp_path / filename_gfits) as g_fits,
        fitsio.FITS(tmp_path / filename_tfits) as t_fits,
    ):
        glass_data = g_fits[1].read()
        test_data = t_fits[1].read()
        assert glass_data["RA"].size == test_data["RA"].size
        assert glass_data["RB"].size == test_data["RA"].size


@pytest.mark.skipif(not HAVE_FITSIO, reason="test requires fitsio")
def test_write_exception(tmp_path: pathlib.Path) -> None:
    class TestWriteError(Exception):
        """Custom exception for controlled testing."""

    def raise_error(msg: str) -> None:
        """Raise a custom exception for controlled testing.

        Parameters
        ----------
        msg
            A message to be passed to the exception.

        Raises
        ------
        TestWriteError
            A custom exception for controlled testing.

        """
        raise TestWriteError(msg)

    try:
        with user.write_catalog(tmp_path / filename, ext="CATALOG") as out:
            for i in range(my_max):
                if i == except_int:
                    msg = "Unhandled exception"
                    raise_error(msg)
                array = np.arange(i, i + 1, delta)  # array of size 1/delta
                array2 = np.arange(i + 1, i + 2, delta)  # array of size 1/delta
                out.write(RA=array, RB=array2)

    except TestWriteError:
        import fitsio

        with fitsio.FITS(tmp_path / filename) as hdul:
            data = hdul[1].read()
            assert data["RA"].size == except_int / delta
            assert data["RB"].size == except_int / delta

            fits_mat = data["RA"].reshape(except_int, int(1 / delta))
            fits_mat2 = data["RB"].reshape(except_int, int(1 / delta))
            for i in range(except_int):
                array = np.arange(
                    i,
                    i + 1,
                    delta,
                )  # re-create array to compare to read data
                array2 = np.arange(i + 1, i + 2, delta)
                assert array.tolist() == fits_mat[i].tolist()
                assert array2.tolist() == fits_mat2[i].tolist()
