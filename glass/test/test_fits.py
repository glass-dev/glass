import pytest

# check if fitsio is available for testing
try:
    import fitsio
except ImportError:
    HAVE_FITSIO = False
else:
    del fitsio
    HAVE_FITSIO = True



'''def _append(filename, data, names):
    import time
    import fitsio
    """Write routine for FITS data."""
    with fitsio.FITS(filename, "rw", clobber=True) as fits:
        fits.write(data, names=names)

@pytest.mark.skipif(not HAVE_FITSIO, reason="test requires fitsio")
def test_basic_write(tmp_path):
    import glass.user as Gfits
    import numpy as np
    d = tmp_path / "sub"
    d.mkdir() 
    filename_gfits = "gfits.fits"
    filename_fits = "fits.fits"
    delta = 0.1 #determines number of points in arrays
    myMax = 100 #target number of threads without exception
    
    with Gfits.awrite(d / filename_gfits, ext="CATALOG") as out:
       for i in range(0,myMax):
           array = np.arange(i, i+1, delta) #array of size 1/delta
           array2 = np.arange(i+1, i+2, delta) #array of size 1/delta
           out.write(RA=array,RB=array2)
           arrays = [array, array2]
           names = ['RA','RB']
           _append(d / filename_fits, arrays, names)   
           
    from astropy.io import fits
    with fits.open(d / filename_gfits) as g_fits, fits.open(d / filename_fits) as my_fits:
          g_data = g_fits[1].data
          my_data = my_fits[1].data
          assert g_data['RA'].size == my_data['RA'].size
          assert g_data['RB'].size == my_data['RA'].size'''
          
filename = "myFile.FITS"      
@pytest.mark.skipif(not HAVE_FITSIO, reason="test requires fitsio")
def test_awrite_exception(tmp_path):
    import glass.user as Gfits
    import numpy as np
    d = tmp_path / "sub"
    d.mkdir() 
    filename = "testAwriteException.fits"
    
    delta = 0.001 #determines number of points in arrays
    myMax = 10000 #target number of threads without exception
    exceptInt = 7500 #where we raise exception in loop
    
    try:
        with Gfits.awrite(d / filename, ext="CATALOG") as out:
            for i in range(0,myMax):
                if i == exceptInt :
                    raise Exception("Unhandled exception") 
                array = np.arange(i, i+1, delta) #array of size 1/delta
                array2 = np.arange(i+1, i+2, delta) #array of size 1/delta
                out.write(RA=array,RB=array2)
                
    except:   
        from astropy.io import fits
        with fits.open(d / filename) as hdul:
            data = hdul[1].data
            assert data['RA'].size == exceptInt/delta
            assert data['RB'].size == exceptInt/delta
            
            fitsMat = data['RA'].reshape(exceptInt,int(1/delta))
            fitsMat2 = data['RB'].reshape(exceptInt,int(1/delta))
            for i in range(0,exceptInt):
                array = np.arange(i, i+1, delta) #re-create array to compare to read data
                array2 = np.arange(i+1, i+2, delta) 
                assert array.tolist() == fitsMat[i].tolist()
                assert array2.tolist() == fitsMat2[i].tolist()        
    
    
@pytest.mark.skipif(not HAVE_FITSIO, reason="test requires fitsio")
def test_out_filename():
    import glass.user as Gfits
    from fitsio import FITS
    fits = FITS(filename, "rw", clobber=True)
    writer = Gfits.AsyncHduWriter(fits)
    assert writer.fits._filename == filename


@pytest.mark.skipif(not HAVE_FITSIO, reason="test requires fitsio")
def test_write_none():
    import glass.user as Gfits
    with Gfits.awrite(filename, ext="CATALOG") as out:
        out.write()
    assert 1 == 1


@pytest.mark.skipif(not HAVE_FITSIO, reason="test requires fitsio")
def test_awrite_yield():
    import glass.user as Gfits
    with Gfits.awrite(filename, ext="CATALOG") as out:
        assert type(out) is Gfits.AsyncHduWriter
