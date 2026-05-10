from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

project = "quantdata"
copyright = "2026, Hans Buehler"
author = "Hans Buehler"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
]

autosummary_generate = True

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

autodoc_typehints = "none"

templates_path = ["_templates"]
exclude_patterns = ["_build", "generated", "Thumbs.db", ".DS_Store"]

html_theme = "alabaster"
