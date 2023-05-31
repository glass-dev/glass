Installation
============

Requirements
------------

*GLASS* works best on a fairly recent version of Python.


The way to install *GLASS* is via pip, which will normally install missing
dependencies as well.  If you are using a package manager such as e.g. conda,
you might want to use that instead to install as many dependencies as possible.
Use ``pip install --dry-run glass`` to list everything that pip would install,
and then see what is available in your package manager.


Install the current release
---------------------------

To install the current *GLASS* release, use the pip package::

    $ pip install glass


Install a specific release
--------------------------

To install the a specific *GLASS* release, pass the version number to pip::

    $ pip install glass==2023.1

For a list of released versions and their respective changes, see the
:doc:`releases`.


Install the latest development version
--------------------------------------

To install the latest *GLASS* code in development, use pip with the GitHub
repository::

    $ pip install git+https://github.com/glass-dev/glass.git


Developer installation
----------------------

If you want to install *GLASS* for local development, clone the repository and
install the package in editable mode::

    $ pip install -e .

All of these words should ideally make sense if you attempt this kind of
installation.


Versioning
----------

There currently are a fair amount of breaking changes between *GLASS* releases.
For production use, it is therefore **strongly** recommended that you install
*GLASS* in a clean environment (venv, conda, etc.) and **pin the GLASS version
for the duration of your project**.


Announcements
-------------

To keep up with new GLASS releases, you can subscribe to our announcement
mailing list, which only receives 1 email per release.  To subscribe, use the
`mailing list page`__, or send an email to ``listserv@jiscmail.ac.uk`` with any
subject and the following message body::

    subscribe glass <Your name>

where ``<Your name>`` is your full name, or ``ANONYMOUS`` if you prefer. You
will be sent a confirmation email in return.

__ https://jiscmail.ac.uk/lists/GLASS.html
