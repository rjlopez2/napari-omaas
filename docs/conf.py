# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

from pathlib import Path

import jupytext
import jupytext.config

sys.path.insert(0, os.path.abspath('../..'))

# -- Convert tutorials -------------------------------------------------------
# convert notebooks to python files for download
def convert_notebooks(chapter_folder):
    tutorials = Path(__file__).parent / chapter_folder
    output_folder = tutorials / "converted"
    output_folder.mkdir(exist_ok=True)

    config = jupytext.config.JupytextConfiguration()
    config.notebook_metadata_filter = "-all"
    config.cell_metadata_filter = "-all"
    
    def remove_input(cell):
        remove_cells_for_tags = ["remove-input"]
        for tag in remove_cells_for_tags:
            if tag in cell.metadata.get("tags", []):
                return False
        return True
    def remove_markdown(cell):
        return cell.cell_type != "markdown"
    
    for path in tutorials.glob("*.ipynb"):
        nb = jupytext.read(path)
        nb['cells'] = list(filter(remove_input, nb.cells))
        jupytext.write(nb, output_folder / f'{path.stem}.ipynb', fmt="ipynb", config=config)
        nb['cells'] = list(filter(remove_markdown, nb.cells))
        jupytext.write(nb, output_folder / f'{path.stem}.py', fmt="py:percent", config=config)
convert_notebooks("tutorials")
convert_notebooks("scripts_examples")


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
nb_execution_excludepatterns = ['*illumination*', '*APD*']
# nb_execution_excludepatterns = [ '*APD*']
nb_execution_timeout = -1
nb_execution_allow_errors=True


# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'tutorials/converted', 'scripts_examples/converted']

# configuration of bibtex_bibfiles settings
bibtex_bibfiles = ['refs.bib']
bibtex_default_style = 'unsrtalpha'
bibtex_reference_style = 'author_year'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_title = 'napari-omaas'
html_theme = "furo"
# html_static_path = ['_static']


autoclass_content = 'both'

python_version = '3.10'

myst_enable_extensions = [
    'colon_fence',
    'dollarmath',
    'substitution',
    'tasklist',
]

# these are variables that can be evaluated in md files
myst_substitutions = {
    "python_version": python_version,
    "conda_create_env": f"```sh\nconda create -y -n {html_title} -c conda-forge python={python_version}\nconda activate {html_title}\n```",
    "under_construction_warn": "```{admonition} 🔨 Work in progress\n:class: warning\nThis tutoriasl is under construction. Soon will be updated.\n```" 
}