import pytest

# check if fitsio is available for testing
try:
    import fitsio
except ImportError:
    HAVE_FITSIO = False
else:
    del fitsio
    HAVE_FITSIO = True

filename = "newfiles.fits"


@pytest.mark.skipif(not HAVE_FITSIO, reason="test requires fitsio")
def test_out_filename():
    import glass.core.fitsIO as GfitsIO
    from fitsio import FITS
    fileToWrite = "myFile.FITS"
    fits = FITS(fileToWrite, "rw", clobber=True)
    writer = GfitsIO.AsyncHduWriter(fits)
    assert writer.fits._filename == fileToWrite


@pytest.mark.skipif(not HAVE_FITSIO, reason="test requires fitsio")
def test_write_none():
    import glass.core.fitsIO as GfitsIO
    with GfitsIO.awrite(filename, ext="CATALOG") as out:
        out.write()
    assert 1 == 1


@pytest.mark.skipif(not HAVE_FITSIO, reason="test requires fitsio")
def test_awrite_yield():
    import glass.core.fitsIO as fitsIO
    with fitsIO.awrite(filename, ext="CATALOG") as out:
        assert type(out) is fitsIO.AsyncHduWriter
