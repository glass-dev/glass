
Examples
========

These examples show how GLASS can be used in practice.  They are often a good
starting point for more complicated and realistic simulations.

The examples currently require `CAMB`__ to produce angular matter power spectra
and for the cosmological background.  Make sure you have CAMB installed:

.. code-block:: console

    $ python -c 'import camb'  # should not give an error

If you want to compute the angular matter power spectra in the examples, you
need the ``glass.ext.camb`` package:

.. code-block:: console

    $ pip install glass.ext.camb

__ https://camb.readthedocs.io/


Basic examples
--------------

To get started, these examples focus on simulating one thing at a time.

.. nbgallery::

   examples/1-basic/shells.ipynb
   examples/1-basic/matter.ipynb
   examples/1-basic/density.ipynb
   examples/1-basic/lensing.ipynb
   examples/1-basic/photoz.ipynb
   examples/1-basic/galaxy-redshift-distributions.ipynb


Advanced examples
-----------------

More advanced examples doing multiple things at the same time.

.. nbgallery::

   examples/2-advanced/cosmic_shear.ipynb
   examples/2-advanced/stage_4_galaxies.ipynb
   examples/2-advanced/legacy-mode.ipynb
