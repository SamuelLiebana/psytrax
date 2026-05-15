"""Sphinx configuration for the psytrax documentation."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

project = "psytrax"
author = "psytrax contributors"
copyright = "2026, psytrax contributors"

extensions = [
    "myst_parser",
    "numpydoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "sphinx_design",
]

autosummary_generate = True
autodoc_typehints = "description"
numpydoc_show_class_members = False
myst_heading_anchors = 3

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "pydata_sphinx_theme"
html_title = "psytrax"
html_static_path = ["_static"]
html_theme_options = {
    "github_url": "https://github.com/SamuelLiebana/psytrax",
    "external_links": [
        {"name": "App", "url": "https://psytrax.streamlit.app"},
    ],
    "show_toc_level": 2,
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "jax": ("https://docs.jax.dev/en/latest/", None),
}
