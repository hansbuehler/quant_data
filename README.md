# quantdata

Pythin environment for systematically processing financial data and building pipelines to feed it to models.
The aim of the project is to joinly develop robust data pipelines which lead to standadized benchmarks in finance.


## Python requirement

Clone the directory ``quant_data``. Change into that directory.

### Create a Python 3.12 environment

This project targets Python 3.12. The repository includes `.python-version` and the package metadata enforces `>=3.12,<3.13`. 
*TODO: make it work an test uner 3.13*

#### Linux/venv

You will need python 3.12. Future versions will support 3.12.
Under Linux, if your defaul Python is not 3.12, you can install a different version with

```bash
sudo apt install python3.12 python3.12-venv  
``` 

Then create a python environment:

```bash
python3.12 -m venv .vqdata # if you need to specify a particular version
python -m venv .vqdata     # otherwise
 
source .vqdata\bin\activate   # activate it
python --version

python -m pip install --upgrade pip pip-review  # update pip
python -m pip install -e .[dev]                 # install local package 
````
Use `.[dev]` for the full local development toolchain. It includes testing and documentation tools.

#### Windows/venv

Under windows you can use the Windows Store to install Python. Once Python 3.12 is installed
create the environment:

```cmd
py -3.12 -m venv .vqdata     # create environemnt
.vqdata\Scripts\Activate     # activate it 
python --version

python -m pip install --upgrade pip pip-review  # update pip
python -m pip install -e .[dev]                 # install local package 
```
Use `.[dev]` for the full local development toolchain. It includes testing and documentation tools.

#### Windows/Anaconda

Using Anaconda, the simplest path is a local conda environment pinned to Python 3.12:

```cmd
conda create --prefix .\.vqdata python=3.12
conda activate .\.vqdata
python --version

python -m pip install --upgrade pip pip-review  # update pip
python -m pip install -e .[dev]                 # install local package 
```
Use `.[dev]` for the full local development toolchain. It includes testing and documentation tools.

If conda plugin warnings interfere with environment creation, retry with:

```cmd
conda create --no-plugins --prefix .\.vqdata python=3.12
```

### Verify the installation

After installation, the package should be importable from the active environment:

```powershell
python -c "import quantdata; print(quantdata.__version__)"
```

## Environment Variables

You will need to set your directory and data access variables.

* ``QUANTDATA_RAW_DATA``: root directory for storing raw cached data. Usually does not need to be fast, e.g. a network drive.
* ``QUANTDATA_PROCESSING_DATA``: intermediate processing: good for multi-process CPU access, but does not need to deliver to GPU.
* ``QUANTDATA_TRAINING_DATA``: training data with the most effective disk to GPU location.

* ``QUANTDATA_ROOT``: if you have everything in one place anyway, use this to set all of the above in one step.

### Caching

Caching in dynamic-sanos is version-aware. You will note that a number of functions have a ``@version``
or ``@cache`` decorator. A functions version often also depends on other function versions. 
A change to the version will trigger a refesh of any cache affected by the respected function - so be careful,
as this can be slow; also do not change the version of a function or class without good reason as it may
trigger refreshs for others once they check out.

* ``QUANTDATA_CACHE_MODE``: is the project-wide caching mode. Use ``off`` to disable, or ``update`` to for an update. Default is ``on``.

### Data Access

* ``MASSIVE_TOKEN`` is the [massive](http://www.massive.com) token.
* ``WRDS_USERNAME``: is the [WRDS](https://wrds-www.wharton.upenn.edu/login/) username.

### Ignore for now

* ``QUANTDATA_TEST_MODE``: global flag to tell the project we are in testing mode. In this mode, no data is downloaded from any data source.
* ``QUANTDATA_TEMP_DATA``: local ephermal caching. Very useful for exceedingly slow processes. Not used at the moment.

## Tests

The pytest test suite enables test mode automatically by setting ``QUANTDATA_TEST_MODE=1`` in
``tests/conftest.py`` before test modules are imported. This avoids access to external data sources during testing.

Run the test suite from the project root with:

```
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
2. Install the test dependencies with `python -m pip install -e .[dev]`.
3. Run `Python: Refresh Tests` from the Command Palette.

Install the Git hook runner and register the local `pytest` pre-commit hook with:

```
python -m pre_commit install
```

GitHub Actions also runs the test suite on pushes to `main` and on pull requests.

## Documentation

Please make an effort to document any files you commit. The project uses [reStructuredText](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html)
which is tedious but the standard. Here is a [comprehensive introduction](https://www.sphinx-doc.org/en/master/).

You build the documentation with
```
python -m sphinx -b html docs docs\_build\html
```

The reStructuredText entry point is `docs/index.rst`.
