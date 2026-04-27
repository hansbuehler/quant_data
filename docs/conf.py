from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

project = "quantdata"
copyright = "2026, Hans Buehler"
author = "Hans Buehler"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "alabaster"
