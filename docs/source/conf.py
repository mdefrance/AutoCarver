import configparser
import os
import sys

# -- Path setup --------------------------------------------------------------
path = "../../"
print(os.listdir(path))
sys.path.insert(0, path)

import AutoCarver

# Read metadata from setup.cfg
config = configparser.ConfigParser()
config.read("../../setup.cfg")  # Provide the correct path to your setup.cfg


# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = config["metadata"]["name"]
copyright = "2023, Mario Defrance"
author = config["metadata"]["author"]
version = config["metadata"]["version"]
release = "Beta Release"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {
    #'canonical_url': '',
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
]
templates_path = ["_templates"]
exclude_patterns = []
autoclass_content = "both"
highlight_language = "python"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
