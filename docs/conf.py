# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

sys.path.insert(0, os.path.abspath(".."))


project = "GATE - Generalization After Transfer Evaluation Engine - An any-modal Model evaluation framework"
copyright = "2023, Antreas Antoniou"
author = "Antreas Antoniou"
release = "0.7.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

language = "python"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
extensions = ["sphinx.ext.autodoc", "sphinx_rtd_theme"]
extensions.append("sphinx.ext.autosummary")
extensions.append("sphinx.ext.viewcode")
extensions.append("sphinx.ext.intersphinx")
extensions.append("sphinx.ext.napoleon")
extensions.append("sphinx.ext.coverage")
extensions.append("sphinx.ext.todo")

# Set to True to show TODOs in the generated documentation
todo_include_todos = True

# Configure the intersphinx_mapping to link to other projects' documentation
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    # Add any other projects you'd like to link to
}
