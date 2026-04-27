# quantdata

Minimal Python package scaffold for the `quantdata` library.

## Python requirement

This project targets Python 3.12. The repository includes `.python-version` and the package metadata enforces `>=3.12,<3.13`.

## Installation

### Create a Python 3.12 environment

On this machine, the simplest path is a local conda environment pinned to Python 3.12:

```powershell
conda create --prefix .\.vqdata python=3.12
conda activate .\.vqdata
python --version
```

If conda plugin warnings interfere with environment creation, retry with:

```powershell
conda create --no-plugins --prefix .\.vqdata python=3.12
```

### Install the package

Upgrade pip and install the project in editable mode:

```powershell
python -m pip install --upgrade pip
python -m pip install -e .
```

### Install optional dependency groups

The project defines optional extras in `pyproject.toml`:

```powershell
python -m pip install -e .[test]
python -m pip install -e .[docs]
python -m pip install -e .[dev]
```

Use `.[dev]` for the full local development toolchain. It includes testing and documentation tools.

### Verify the installation

After installation, the package should be importable from the active environment:

```powershell
python -c "import quantdata; print(quantdata.__version__)"
```

## Tests

The pytest test suite enables test mode automatically by setting `QUANTDATA_TEST_MODE=1` in `tests/conftest.py` before test modules are imported. That bootstrap also creates the local `tests/stored_data/raw`, `tests/stored_data/processing`, and `tests/stored_data/training` directories if they do not already exist.

Run the test suite with:

```powershell
python -m pytest
```

If you want to reproduce the same behavior manually in a terminal, run:

```powershell
$env:QUANTDATA_TEST_MODE = "1"
python -m pytest
```

### Run tests in VS Code

1. Open the repository folder in VS Code.
2. Select the Python interpreter for the workspace.
3. Use the local `.vqdata` environment once it has been created with Python 3.12.
4. Open the Command Palette and run `Python: Select Interpreter` if needed.
5. Open the Testing view in the left sidebar.
6. If VS Code prompts to configure Python tests, choose `pytest`.
7. When it asks for the test folder, choose `tests`.

The workspace already includes `.vscode/settings.json`, so VS Code should use `pytest` with the `tests` directory automatically.

After configuration, you can run tests in any of these ways:

```text
Testing sidebar -> Run All Tests
Command Palette -> Python: Run All Tests
Editor gutter -> Run Test on an individual test function
Terminal -> python -m pytest
```

If test discovery is empty, the usual fix is:

1. Confirm the selected interpreter is the `.vqdata` environment.
2. Install the test dependencies with `python -m pip install -e .[test]` or `python -m pip install -e .[dev]`.
3. Run `Python: Refresh Tests` from the Command Palette.

Install the Git hook runner and register the local `pytest` pre-commit hook with:

```powershell
python -m pre_commit install
```

GitHub Actions also runs the test suite on pushes to `main` and on pull requests.

## Documentation

Build the Sphinx documentation with:

```powershell
python -m sphinx -b html docs docs\_build\html
```

The reStructuredText entry point is `docs/index.rst`.