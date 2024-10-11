# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


# -- Project information -----------------------------------------------------

import datetime
from importlib import metadata

project = "GLASS"
author = "GLASS developers"
copyright = f"2022-{datetime.date.today().year} {author}"  # noqa: A001, DTZ011
version = metadata.version("glass")
release = version


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "matplotlib.sphinxext.plot_directive",
    "nbsphinx",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinxcontrib.katex",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages. See the documentation for
# a list of builtin themes.
#
html_theme = "furo"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_logo = "_static/logo.png"
html_favicon = "_static/favicon.ico"
html_css_files = []


# -- Intersphinx -------------------------------------------------------------

# This config value contains the locations and names of other projects that
# should be linked to in this documentation.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}


# -- autodoc -----------------------------------------------------------------

autodoc_typehints = "description"


# -- napoleon ----------------------------------------------------------------

napoleon_google_docstring = False
napoleon_use_rtype = False


# -- plot_directive ----------------------------------------------------------

# Whether to show a link to the source in HTML (default: True).
plot_html_show_source_link = False

# Whether to show links to the files in HTML (default: True).
plot_html_show_formats = False

# File formats to generate (default: ['png', 'hires.png', 'pdf']).
plot_formats = [("svg", 150), ("png", 150)]

# A dictionary containing any non-standard rcParams that should be applied
# before each plot (default: {}).
plot_rcparams = {
    "axes.facecolor": (1.0, 1.0, 1.0, 1.0),
    "savefig.facecolor": (1.0, 1.0, 1.0, 0.5),
    "savefig.transparent": False,
}
