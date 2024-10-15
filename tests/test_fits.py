import importlib.util
import os
import pathlib

import numpy as np
import pytest

from glass import user

# check if fitsio is available for testing
HAVE_FITSIO = importlib.util.find_spec("fitsio") is not None


def _test_append(fits, data, names) -> None:  # type: ignore[no-untyped-def]
    """Write routine for FITS test cases."""
    cat_name = "CATALOG"
    if cat_name not in fits:
        fits.write_table(data, names=names, extname=cat_name)
    else:
        hdu = fits[cat_name]
        hdu.write(data, names=names, firstrow=hdu.get_nrows())


delta = 0.001  # Number of points in arrays
my_max = 1000  # Typically number of galaxies in loop
except_int = 750  # Where test exception occurs in loop
filename = "MyFile.Fits"


@pytest.mark.skipif(not HAVE_FITSIO, reason="test requires fitsio")
def test_basic_write(tmp_path: os.PathLike) -> None:  # type: ignore[type-arg]
    import fitsio

    filename_gfits = "gfits.fits"  # what GLASS creates
    filename_tfits = "tfits.fits"  # file created on the fly to test against

    with (
        user.write_catalog(tmp_path / filename_gfits, ext="CATALOG") as out,  # type: ignore[operator]
        fitsio.FITS(tmp_path / filename_tfits, "rw", clobber=True) as my_fits,  # type: ignore[operator]
    ):
        for i in range(my_max):
            array = np.arange(i, i + 1, delta)  # array of size 1/delta
            array2 = np.arange(i + 1, i + 2, delta)  # array of size 1/delta
            out.write(RA=array, RB=array2)
            arrays = [array, array2]
            names = ["RA", "RB"]
            _test_append(my_fits, arrays, names)  # type: ignore[no-untyped-call]

    with (
        fitsio.FITS(tmp_path / filename_gfits) as g_fits,  # type: ignore[operator]
        fitsio.FITS(tmp_path / filename_tfits) as t_fits,  # type: ignore[operator]
    ):
        glass_data = g_fits[1].read()
        test_data = t_fits[1].read()
        assert glass_data["RA"].size == test_data["RA"].size
        assert glass_data["RB"].size == test_data["RA"].size


@pytest.mark.skipif(not HAVE_FITSIO, reason="test requires fitsio")
def test_write_exception(tmp_path: pathlib.Path) -> None:
    try:
        with user.write_catalog(tmp_path / filename, ext="CATALOG") as out:  # type: ignore[arg-type]
            for i in range(my_max):
                if i == except_int:
                    msg = "Unhandled exception"
                    raise Exception(msg)  # noqa: TRY002, TRY301
                array = np.arange(i, i + 1, delta)  # array of size 1/delta
                array2 = np.arange(i + 1, i + 2, delta)  # array of size 1/delta
                out.write(RA=array, RB=array2)

    except Exception:  # noqa: BLE001
        import fitsio

        with fitsio.FITS(tmp_path / filename) as hdul:
            data = hdul[1].read()
            assert data["RA"].size == except_int / delta
            assert data["RB"].size == except_int / delta

            fitsMat = data["RA"].reshape(except_int, int(1 / delta))  # noqa: N806
            fitsMat2 = data["RB"].reshape(except_int, int(1 / delta))  # noqa: N806
            for i in range(except_int):
                array = np.arange(
                    i,
                    i + 1,
                    delta,
                )  # re-create array to compare to read data
                array2 = np.arange(i + 1, i + 2, delta)
                assert array.tolist() == fitsMat[i].tolist()
                assert array2.tolist() == fitsMat2[i].tolist()
