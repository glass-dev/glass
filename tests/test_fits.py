import pytest

# check if fitsio is available for testing
import importlib.util
if importlib.util.find_spec("fitsio") is not None:
    HAVE_FITSIO = True
else:
    HAVE_FITSIO = False

import glass.user as user
import numpy as np


def _test_append(fits, data, names):
    '''Write routine for FITS test cases'''
    cat_name = 'CATALOG'
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
def test_basic_write(tmp_path):
    import fitsio
    d = tmp_path / "sub"
    d.mkdir()
    filename_gfits = "gfits.fits"  # what GLASS creates
    filename_tfits = "tfits.fits"  # file created on the fly to test against

    with user.write_catalog(d / filename_gfits, ext="CATALOG") as out, fitsio.FITS(d / filename_tfits, "rw", clobber=True) as myFits:
        for i in range(0, my_max):
            array = np.arange(i, i+1, delta)  # array of size 1/delta
            array2 = np.arange(i+1, i+2, delta)  # array of size 1/delta
            out.write(RA=array, RB=array2)
            arrays = [array, array2]
            names = ['RA', 'RB']
            _test_append(myFits, arrays, names)

    with fitsio.FITS(d / filename_gfits) as g_fits, fitsio.FITS(d / filename_tfits) as t_fits:
        glass_data = g_fits[1].read()
        test_data = t_fits[1].read()
        assert glass_data['RA'].size == test_data['RA'].size
        assert glass_data['RB'].size == test_data['RA'].size


@pytest.mark.skipif(not HAVE_FITSIO, reason="test requires fitsio")
def test_write_exception(tmp_path):
    d = tmp_path / "sub"
    d.mkdir()

    try:
        with user.write_catalog(d / filename, ext="CATALOG") as out:
            for i in range(0, my_max):
                if i == except_int:
                    raise Exception("Unhandled exception")
                array = np.arange(i, i+1, delta)  # array of size 1/delta
                array2 = np.arange(i+1, i+2, delta)  # array of size 1/delta
                out.write(RA=array, RB=array2)

    except Exception:
        import fitsio
        with fitsio.FITS(d / filename) as hdul:
            data = hdul[1].read()
            assert data['RA'].size == except_int/delta
            assert data['RB'].size == except_int/delta

            fitsMat = data['RA'].reshape(except_int, int(1/delta))
            fitsMat2 = data['RB'].reshape(except_int, int(1/delta))
            for i in range(0, except_int):
                array = np.arange(i, i+1, delta)  # re-create array to compare to read data
                array2 = np.arange(i+1, i+2, delta)
                assert array.tolist() == fitsMat[i].tolist()
                assert array2.tolist() == fitsMat2[i].tolist()


@pytest.mark.skipif(not HAVE_FITSIO, reason="test requires fitsio")
def test_out_filename(tmp_path):
    import fitsio
    fits = fitsio.FITS(filename, "rw", clobber=True)
    writer = user._FitsWriter(fits)
    assert writer.fits._filename == filename
