
Getting started
===============

Installation
------------

GLASS requires a fairly recent version of Python.

To install the latest GLASS release, use the pip package::

    $ pip install glass

To install the latest GLASS code in development, use pip with the GitHub
repository::

    $ pip install git+https://github.com/glass-dev/glass.git

As usual, pip will install missing dependencies as well.  If you are using a
package manager such as e.g. conda, you might want to use that to install as
many of the packages as possible.  Use ``pip install --dry-run glass`` to see
all missing dependencies.

If you want to install GLASS for local development, clone the repository and
install the package in editable mode with ``pip install -e .`` -- all of these
words should ideally make sense if you attempt that.

For production use, it is **strongly** recommended that you install GLASS in a
clean environment (venv, conda, etc.) and **pin the GLASS version for the
duration of the project**.


First steps
-----------

The best way to get started with GLASS is to follow the examples__.

__ https://glass.readthedocs.io/projects/examples/


Getting in touch
----------------

If you would like to start a discussion with the wider GLASS community about
e.g. a design decision or API change, you can use our Discussions__ page.

__ https://github.com/orgs/glass-dev/discussions

We also have a public `Slack workspace`__ for discussions about the project.

__ https://glass-dev.github.io/slack
