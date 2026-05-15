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
myst_enable_extensions = ["colon_fence"]
myst_heading_anchors = 3

templates_path = ["_templates"]
exclude_patterns = ["snippets/*"]

html_theme = "pydata_sphinx_theme"
html_title = "psytrax"
html_logo = "_static/psytrax_logo.svg"
html_static_path = ["_static"]
html_css_files = ["psytrax.css"]
html_theme_options = {
    "logo": {
        "image_light": "_static/psytrax_logo.svg",
        "image_dark": "_static/psytrax_logo_dark.svg",
    },
    "navbar_start": ["navbar-logo"],
    "navbar_center": [],
    "navbar_end": [],
    "navbar_persistent": [],
    "icon_links": [
        {
            "name": "Zulip",
            "url": "https://neuroinformatics.zulipchat.com",
            "icon": "fa-solid fa-comments",
            "type": "fontawesome",
        },
    ],
    "github_url": "https://github.com/SamuelLiebana/psytrax",
    "external_links": [
        {"name": "App", "url": "https://psytrax.streamlit.app"},
    ],
    "footer_start": ["footer-logos"],
    "footer_center": ["copyright"],
    "footer_end": ["sphinx-version"],
    "show_toc_level": 2,
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "jax": ("https://docs.jax.dev/en/latest/", None),
}
