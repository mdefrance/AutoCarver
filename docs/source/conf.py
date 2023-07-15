import os
import sys

# -- Path setup --------------------------------------------------------------
print(os.listdir('../../AutoCarver'))
sys.path.append('../../AutoCarver')
# sys.path.insert(0, os.path.abspath('../../AutoCarver'))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'AutoCarver'
copyright = '2023, Mario Defrance'
author = 'Mario Defrance'
release = 'AutoCarver 5.0.7 - Beta Release'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx_rtd_theme',
    'sphinx.ext.autodoc',
    #"sphinx.ext.autosummary",
]

templates_path = ['_templates']
exclude_patterns = []

highlight_language = 'python'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
