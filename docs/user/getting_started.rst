
Getting started
===============

Installation
------------

First steps
-----------

Using the script runner
-----------------------
GLASS code can be run like any other Python code, either as part of a bigger
simulation pipeline, or as a stand-alone script.  For the latter case, the core
library provides a wrapper that runs scripts from the command line:

.. code-block:: console

    $ python -m glass --help
    usage: python -m glass [-h] [-l LOGLEVEL] [-D] path ...

This sets up logging, but otherwise runs any script it is given as it is, using
the :mod:`runpy` standard library module.

The script runner is particularly useful if called with the ``-D`` flag, which
starts an interactive debugging session when an unhandled exception is raised:

.. code-block:: console

    $ python -m glass -D my_simulation.py --my_flag ...

The debugger would be started at the correct location inside a generator if it
caused the exception.


Questions and answers
---------------------

For Q&As about GLASS itself, please use the Discussions__ page within the main
GLASS GitHub repository.

__ https://github.com/astro-ph/glass/discussions 

For Q&As about GLASS extensions, please use the respective Discussions pages
within the extensions' GitHub repositories.
