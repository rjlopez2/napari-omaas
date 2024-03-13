# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'napari-omaas'
copyright = '2024, Ruben Lopez'
author = 'Ruben Lopez'
release = '0.1.3'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinx_codeautolink",
    'sphinxcontrib.bibtex',
    "sphinxcontrib.video",
    "sphinx.ext.viewcode",
    "myst_nb",
]

templates_path = ['_templates']
exclude_patterns = []


# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'tutorials/converted']

# configuration of bibtex_bibfiles settings
bibtex_bibfiles = ['refs.bib']
bibtex_default_style = 'unsrtalpha'
bibtex_reference_style = 'author_year'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_title = "napari-omaas"
html_theme = "furo"
html_static_path = ['_static']


autoclass_content = 'both'