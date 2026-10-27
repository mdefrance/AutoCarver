"""Config for sphinx documentation"""

import os
import sys
from inspect import getsourcefile
from pathlib import Path

import toml
from sphinx.util.nodes import make_refnode


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
DOCS_DIRECTORY = Path(getsourcefile(lambda: 0)).resolve().parent


def ensure_pandoc_installed(_):
    import pypandoc

    # Download pandoc if necessary. If pandoc is already installed and on
    # the PATH, the installed version will be used. Otherwise, we will
    # download a copy of pandoc into docs/bin/ and add that to our PATH.
    pandoc_dir = DOCS_DIRECTORY / "bin"
    # Add dir containing pandoc binary to the PATH environment variable
    if pandoc_dir not in os.environ["PATH"].split(os.pathsep):
        os.environ["PATH"] += os.pathsep + pandoc_dir.as_posix()
    pypandoc.ensure_pandoc_installed(
        # quiet=True,
        targetfolder=pandoc_dir,
        delete_installer=True,
    )


def setup(app):
    """ensures that pandoc is installed, links unqualified AutoCarver names"""
    app.connect("builder-inited", ensure_pandoc_installed)
    app.connect("missing-reference", _resolve_short_names, priority=400)  # before intersphinx


# Read metadata from setup.cfg
pyproject = toml.load("../../pyproject.toml")


# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = pyproject["project"]["name"]
copyright = "2023–2026, Mario Defrance"
author = ", ".join([author["name"] for author in pyproject["project"]["authors"]])
version = pyproject["project"]["version"]
release = version

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
# Dark (light-ink) variant: the RTD theme's sidebar header background is dark.
html_logo = "artwork/auto_carver_symbol_dark.svg"


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_rtd_theme",
    # "sphinx.ext.autosummary",
    "sphinx_design",
    "nbsphinx",
]

# links to the docs of the types used in signatures and docstrings
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
}

# docstring type words that are prose, not objects (numpydoc "int, optional", sklearn's "array-like of shape")
nitpick_ignore = [
    ("py:class", word) for word in ("optional", "array-like", "shape", "ndarray array", "scalar", "iterable", "numbers")
]
nitpick_ignore_regex = [
    ("py:class", r"default.*|.*n_samples.*|.*n_features.*|.*n_outputs.*|.*cross-validation generator|\".*\"")
]

# short names used in docstrings, resolved by intersphinx once rewritten
_TYPE_ALIASES = {
    "DataFrame": "pandas.DataFrame",
    "Series": "pandas.Series",
    "ndarray": "numpy.ndarray",
    "Path": "pathlib.Path",
}
_MODULE_ALIASES = {"pd": "pandas", "np": "numpy"}


def _resolve_short_names(app, env, node, contnode):
    """Links an unqualified AutoCarver name (``Features``, ``BinaryCarver``) to its documented object
    when exactly one documented object has that name, and rewrites ``pd.``/``np.`` names for intersphinx."""
    if node.get("refdomain") != "py":
        return None
    target = node["reftarget"]
    module, _, rest = target.partition(".")
    if module in _MODULE_ALIASES and rest:
        node["reftarget"] = f"{_MODULE_ALIASES[module]}.{rest}"
        return None
    if target in _TYPE_ALIASES:
        node["reftarget"] = _TYPE_ALIASES[target]
        return None
    name = target.lstrip("~.").split(".")[-1]
    domain = env.get_domain("py")
    found = {(obj.docname, obj.node_id) for full, obj in domain.objects.items() if full.split(".")[-1] == name}
    if len(found) != 1:
        return None
    docname, node_id = found.pop()
    return make_refnode(app.builder, node["refdoc"], docname, node_id, contnode, name)


templates_path = ["_templates"]
# the ProTips notebooks are kept in the repo but not published
exclude_patterns = ["examples/.ipynb_checkpoints", "examples/Carvers/ProTips"]
# notebooks are rendered from their stored outputs (executed with nbconvert before each docs release)
nbsphinx_execute = "never"
autoclass_content = "both"
highlight_language = "python"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
