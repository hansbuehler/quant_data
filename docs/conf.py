import importlib
import os
from pathlib import Path
import re
import sys

from sphinx.domains.python import PyXRefRole
from sphinx.util import logging as sphinx_logging

DOCS_ROOT = Path(__file__).resolve().parent
REPO_ROOT = DOCS_ROOT.parent
PACKAGE_ROOT = REPO_ROOT / "quantdata"
GENERATED_API_DIR = DOCS_ROOT / "api_generated"
DOCS_BUILD_ROOT = DOCS_ROOT / "_build" / "quantdata-root"
LOGGER = sphinx_logging.getLogger(__name__)

sys.path.insert(0, str(REPO_ROOT))

# Make docs generation reproducible in CI and on fresh machines where the
# quantdata cache roots are not configured.
DOCS_BUILD_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("QUANTDATA_ROOT", str(DOCS_BUILD_ROOT))

project = "quantdata"
copyright = "2026, Hans Buehler"
author = "Hans Buehler"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "numpydoc",
    "sphinx_automodapi.automodapi",
    "sphinx_copybutton",
    "sphinx_design",
    "myst_parser",
]

autosummary_generate = True

autodoc_default_options = {
    "members": True,
    "inherited-members": False,
    "undoc-members": False,
    "show-inheritance": False,
    "special-members": "__call__",
}

autodoc_typehints = "signature"
autodoc_member_order = "bysource"
typehints_document_rtype = False
numpydoc_show_class_members = False
numpydoc_class_members_toctree = False

templates_path = ["_templates"]
exclude_patterns = ["_build", "generated", "Thumbs.db", ".DS_Store"]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "navigation_depth": 2,
    "show_prev_next": False,
    "github_url": "https://github.com/hansbuehler/quant_data",
    "show_toc_level": 3,
    "secondary_sidebar_items": ["page-toc", "sourcelink"],
}
html_static_path = ["_static"]
html_css_files = ["custom.css"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "ipython": ("https://ipython.readthedocs.io/en/stable/", None),
}

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "amsmath",
    "linkify",
]

_LITERAL_DOCSTRING_NAMES = {
    "quantdata.biz_calendar.NMS",
    "quantdata.sanos",
}

_DOCSTRING_SUMMARY_OVERRIDES = {
    "quantdata.dysanos.simple.model.AttentionPoolingEncoderFactory": "Attention pooling encoder factory.",
}

_LITERAL_DOCSTRING_PREFIXES = (
    "quantdata.dysanos.simple.model.AttentionPoolingEncoderFactory",
    "quantdata.dysanos.simple.sampler",
    "quantdata.massive.client.MassiveClient",
    "quantdata.massive.options.contracts",
    "quantdata.sanos.grid",
    "quantdata.utils.sanos.MkArbFree",
    "quantdata.widb.options_filter.OptionsFilter",
)

_DOCSTRING_ROLE_RE = re.compile(r":[a-zA-Z]+:`([^`]+)`")
_DOCSTRING_LINK_RE = re.compile(r"`([^`<>]+?)\s*<\s*(https?://[^>]+)\s*>`_?")
_DOCSTRING_HEADING_RE = re.compile(r"\s*[\^=\-`~#*]{3,}\s*")

_PACKAGE_DIR_EXCLUDES = {"__pycache__", "notebooks"}


def _safe_import_module(module_name: str):
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - exercised by Sphinx builds
        LOGGER.info("Skipping API docs for %s: %s", module_name, exc)
        return None

    module_file = getattr(module, "__file__", None)
    if module_file is None:
        return module

    module_path = Path(module_file).resolve()
    if not module_path.is_relative_to(PACKAGE_ROOT):
        LOGGER.info(
            "Skipping API docs for %s because it resolves outside the quantdata tree: %s",
            module_name,
            module_path,
        )
        return None

    return module


def _iter_package_children(package_name: str, package_path: Path):
    packages = []
    modules = []

    for child in sorted(package_path.iterdir(), key=lambda path: path.name):
        if child.name.startswith("_"):
            continue

        if child.is_dir():
            if child.name in _PACKAGE_DIR_EXCLUDES or not (child / "__init__.py").exists():
                continue

            module_name = f"{package_name}.{child.name}"
            if _safe_import_module(module_name) is None:
                continue
            packages.append((module_name, child))
            continue

        if not child.is_file() or child.suffix != ".py" or child.name == "__init__.py":
            continue

        module_name = f"{package_name}.{child.stem}"
        if _safe_import_module(module_name) is None:
            continue
        modules.append(module_name)

    return packages, modules


def _write_module_page(module_name: str, output_dir: Path) -> None:
    lines = [
        module_name,
        "=" * len(module_name),
        "",
        f".. automodule:: {module_name}",
        "   :members:",
        "   :show-inheritance:",
        "",
    ]

    (output_dir / f"{module_name}.rst").write_text("\n".join(lines), encoding="utf-8")


def _write_package_page(package_name: str, package_path: Path, output_dir: Path) -> None:
    child_packages, child_modules = _iter_package_children(package_name, package_path)

    lines = [package_name, "=" * len(package_name), ""]

    if package_name != "quantdata":
        lines.extend(
            [
                f".. automodule:: {package_name}",
                "   :members:",
                "   :show-inheritance:",
                "",
            ]
        )

    if child_packages:
        lines.extend(["Subpackages", "-----------", "", ".. toctree::", "   :maxdepth: 1", ""])
        for child_package_name, _ in child_packages:
            lines.append(f"   {child_package_name}")
        lines.append("")

    if child_modules:
        lines.extend(["Modules", "-------", "", ".. toctree::", "   :maxdepth: 1", ""])
        for child_module_name in child_modules:
            lines.append(f"   {child_module_name}")
        lines.append("")

    (output_dir / f"{package_name}.rst").write_text("\n".join(lines), encoding="utf-8")

    for child_package_name, child_package_path in child_packages:
        _write_package_page(child_package_name, child_package_path, output_dir)

    for child_module_name in child_modules:
        _write_module_page(child_module_name, output_dir)


def _generate_api_pages(app) -> None:
    GENERATED_API_DIR.mkdir(exist_ok=True)
    for rst_file in GENERATED_API_DIR.glob("*.rst"):
        rst_file.unlink()

    _write_package_page("quantdata", PACKAGE_ROOT, GENERATED_API_DIR)


def _plain_doc_line(line: str) -> str:
    stripped = line.strip()
    if stripped.startswith("```") or stripped == "$$":
        return ""

    line = _DOCSTRING_LINK_RE.sub(r"\1 (\2)", line)
    line = _DOCSTRING_ROLE_RE.sub(r"\1", line)
    line = line.replace("``", "")
    line = line.replace("`", "")
    line = line.replace("|", "")
    return line.rstrip()


def _literalize_docstring(name: str, lines: list[str]) -> None:
    content = [line.rstrip() for line in lines]
    while content and not content[0].strip():
        content.pop(0)
    while content and not content[-1].strip():
        content.pop()

    if not content:
        return

    summary = _plain_doc_line(content[0]).strip()
    body_start = 1

    if len(content) > 1 and _DOCSTRING_HEADING_RE.fullmatch(content[1]):
        for index in range(2, len(content)):
            candidate = _plain_doc_line(content[index]).strip()
            if candidate:
                summary = candidate
                body_start = index + 1
                break

    if not summary:
        summary = name.rsplit(".", 1)[-1]

    body = []
    for line in content[body_start:]:
        if _DOCSTRING_HEADING_RE.fullmatch(line):
            continue
        body.append(_plain_doc_line(line))

    while body and not body[0].strip():
        body.pop(0)
    while body and not body[-1].strip():
        body.pop()

    if not body:
        lines[:] = [summary]
        return

    literalized = [summary, "", "::", ""]
    for line in body:
        literalized.append(f"   {line}" if line else "")
    lines[:] = literalized


def _normalize_problem_docstrings(app, what, name, obj, options, lines):
    if name in _DOCSTRING_SUMMARY_OVERRIDES:
        lines[:] = [_DOCSTRING_SUMMARY_OVERRIDES[name]]
        return

    if name in _LITERAL_DOCSTRING_NAMES or any(
        name.startswith(prefix) for prefix in _LITERAL_DOCSTRING_PREFIXES
    ):
        _literalize_docstring(name, lines)


def _skip_problem_members(app, what, name, obj, skip, options):
    if name == "Tensor":
        return True

    member_module = getattr(obj, "__module__", None)
    if isinstance(member_module, str) and member_module and not member_module.startswith("quantdata"):
        return True

    module_name = getattr(obj, "__name__", None)
    if isinstance(obj, type(sys)) and isinstance(module_name, str) and not module_name.startswith("quantdata"):
        return True

    return skip


def setup(app):
    app.connect("builder-inited", _generate_api_pages)
    app.connect("autodoc-process-docstring", _normalize_problem_docstrings)
    app.connect("autodoc-skip-member", _skip_problem_members)
    app.add_role_to_domain("py", "dec", PyXRefRole("func"))
    return {"parallel_read_safe": True}
