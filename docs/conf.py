# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('..'))


# -- Project information -----------------------------------------------------

project = 'GLASS'
copyright = '2022, Nicolas Tessore'
author = 'Nicolas Tessore'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'numpydoc',
    'sphinx.ext.intersphinx',
    'sphinxcontrib.mermaid',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'pydata_sphinx_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_logo = '_static/glass.png'

html_favicon = '_static/glass.ico'

html_theme_options = {
    'logo': {
        'text': project,
    },
    'external_links': [
        {
            'name': 'Examples',
            'url': 'https://glass.readthedocs.io/projects/examples/',
        },
    ],
    'icon_links': [
        {
            'name': 'GitHub',
            'url': 'https://github.com/glass-dev/glass',
            'icon': 'fab fa-github',
            'type': 'fontawesome',
        },
    ],
}

html_css_files = [
    'css/custom.css',
]


# -- Intersphinx -------------------------------------------------------------

# This config value contains the locations and names of other projects that
# should be linked to in this documentation.
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'examples': ('https://glass.readthedocs.io/projects/examples/en/latest/', None),
}


# -- numpydoc ----------------------------------------------------------------

# Whether to create a Sphinx table of contents for the lists of class methods
# and attributes. If a table of contents is made, Sphinx expects each entry to
# have a separate page.
numpydoc_class_members_toctree = False
