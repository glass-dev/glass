.. module:: glass.healpix

:mod:`glass.healpix` --- Array API compatible HEALPix functions
===============================================================

.. currentmodule:: glass.healpix

This module contains functions for working with HEALPix maps in an Array API
compliant manner used with GLASS.

This module should be imported manually if used outside of GLASS::

    import glass.healpix as hp

``healpix`` Functions
---------------------

.. autofunction:: ang2pix
.. autofunction:: ang2vec
.. autofunction:: npix2nside
.. autofunction:: nside2npix
.. autofunction:: randang

``healpy`` Functions
--------------------

.. autofunction:: alm2map
.. autofunction:: alm2map_spin
.. autofunction:: almxfl
.. autofunction:: get_nside
.. autofunction:: map2alm
.. autofunction:: pixwin
.. autofunction:: query_strip
.. automethod:: Rotator.rotate_map_pixel
