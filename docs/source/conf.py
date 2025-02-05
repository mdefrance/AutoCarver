""" Config for sphinx documentation
"""

import os
import sys
from inspect import getsourcefile

import toml


# -- Path setup --------------------------------------------------------------
def setup_path_and_check_autocarver():
    """Setup path to AutoCarver and check if it is installed"""
    path = "../../"
    sys.path.insert(0, path)

    import AutoCarver

    _ = AutoCarver


setup_path_and_check_autocarver()

# make copy of notebooks in docs folder, as they must be here for sphinx to
# pick them up properly.
# NOTEBOOKS_DIR = os.path.abspath("example_notebooks")
# if os.path.exists(NOTEBOOKS_DIR):
#     import warnings

#     warnings.warn("example_notebooks directory exists, replacing...")
#     shutil.rmtree(NOTEBOOKS_DIR)
# shutil.copytree(os.path.abspath("./examples"), NOTEBOOKS_DIR)
# if os.path.exists(NOTEBOOKS_DIR + "/local_scratch"):
#     shutil.rmtree(NOTEBOOKS_DIR + "/local_scratch")


# Get path to directory containing this file, conf.py.
DOCS_DIRECTORY = os.path.dirname(os.path.abspath(getsourcefile(lambda: 0)))


def ensure_pandoc_installed(_):
    import pypandoc

    # Download pandoc if necessary. If pandoc is already installed and on
    # the PATH, the installed version will be used. Otherwise, we will
    # download a copy of pandoc into docs/bin/ and add that to our PATH.
    pandoc_dir = os.path.join(DOCS_DIRECTORY, "bin")
    # Add dir containing pandoc binary to the PATH environment variable
    if pandoc_dir not in os.environ["PATH"].split(os.pathsep):
        os.environ["PATH"] += os.pathsep + pandoc_dir
    pypandoc.ensure_pandoc_installed(
        # quiet=True,
        targetfolder=pandoc_dir,
        delete_installer=True,
    )


def setup(app):
    """ensures that pandoc is installed"""
    app.connect("builder-inited", ensure_pandoc_installed)


# Read metadata from setup.cfg
pyproject = toml.load("../../pyproject.toml")


# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = pyproject["tool"]["poetry"]["name"]
copyright = "2023, Mario Defrance"
author = pyproject["tool"]["poetry"]["authors"][0]
version = pyproject["tool"]["poetry"]["version"]
release = "Beta release"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {
    # 'canonical_url': '',
    "logo_only": True,
    "display_version": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    # "style_nav_header_background": "#343131",
    # Toc options
    "collapse_navigation": True,
    "sticky_navigation": True,
    "navigation_depth": 2,
    "includehidden": True,
    "titles_only": False,
}

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = "artwork/auto_carver_symbol_small.png"


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_rtd_theme",
    "sphinx.ext.autodoc",
    # "sphinx.ext.autosummary",
    "nbsphinx",
]
templates_path = ["_templates"]
exclude_patterns = []
autoclass_content = "both"
highlight_language = "python"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
