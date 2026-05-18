# quantdata

Python environment for systematically processing financial data and building pipelines to feed it to models.
The aim of the project is to joinly develop robust data pipelines which lead to standadized benchmarks in quantitative finance.

Principles:
* [Data Handling Pricinple & Guidelines](DATA-MANAGEMENT.md)
* [Conding Guidelines](CODE.MD) [not written yet]
* [Model Benchmarking](BENCHMARKING.md) [not written yet]

# Setup

You will need to
1) Install the environment
2) Use environment variables to set your caching directories.
3) Register your license for each data source.

# Documentation

Code generated docuemntation is still in progress; it can be found here [https://hansbuehler.github.io/quant_data](https://hansbuehler.github.io/quant_data/).

## TL;DR

Once cloned, enter into the ``quant_data`` directory and install the environment:

```bash
python -m pip install -U pip pip-review 
python -m pip install -U -e ".[dev]"              
```

For no-frills functionality set

```bash
export QUANTDATA_ROOT=~/quantdata_data/
export WRDS_USERNAME=<wrds_username>
export MASSIVE_TOKEN=<massive_token>
```

Run e.g. ``quantdata/widb/viz_day.ipynb`` to test WRDS Option Metrics or ``quantdata/massive/viz_bars.ipynb`` for a Massive example.

You can set more granular data directories, [see below](#data-locations).

## Creating a Python Environment

This project targets Python 3.12 but should run under 3.13.

#### Linux/venv

Under Linux, if your default Python is not 3.12, you can install a different version with

```bash
sudo apt update
sudo apt install python3.12 python3.12-venv  
``` 

Then create a python environment:
```bash
python3.12 -m venv .vqdata 
```
or
```bash
python -m venv .vqdata   
```
 
Then activate and install dependencies:
 ```bash
source .vqdata/bin/activate   # activate it
python --version

python -m pip install -U pip pip-review 
python -m pip install -U -e ".[dev]"              
````
Use `.[dev]` for the full local development toolchain. It includes testing and documentation tools.

#### Windows/venv

Under windows you can use the Windows Store to install Python. Once Python 3.12 is installed
create the environment:

```cmd
py -3.12 -m venv .vqdata     
```
or, if you have only one python installed,
```cmd
python -m venv .vqdata     
```
Then follow up with
```cmd
.vqdata\Scripts\Activate     
python --version

python -m pip install --upgrade pip pip-review  
python -m pip install -U -e .[dev]                 
```
Use `.[dev]` for the full local development toolchain. It includes testing and documentation tools.

#### Windows/Anaconda

Using Anaconda, the simplest path is a local conda environment pinned to Python 3.12:

```cmd
conda create --prefix .\.vqdata python=3.12
conda activate .\.vqdata
python --version

python -m pip install -U pip pip-review  
python -m pip install -U -e .[dev]                
```
Use `.[dev]` for the full local development toolchain. It includes testing and documentation tools.

If conda plugin warnings interfere with environment creation, retry with:

```cmd
conda create --no-plugins --prefix .\.vqdata python=3.12
```

#### uv

If you prefer `uv`, create the local environment with

```bash
uv sync --extra dev
source .venv/bin/activate
python --version
```

This will create a local `.venv` and install the project with the full development toolchain.
If you do not want to activate the environment, you can run commands with `uv run`, e.g. `uv run pytest`.

### Verify the installation

After installation, the package should be importable from the active environment:

```powershell
python -c "import quantdata; print(quantdata.__version__)"
```

## Environment Variables

The project environment is organised around two main inputs: data storage/caching locations on one hand side, and access tokens to data sources on the other.

### Data Locations

The project is organized around proactive caching of:

* **raw data**: downloaded from a data source. Such data can be stored in slower drives. The point of the cache is to reduce download overhead and intensity.
* **processed data**: enriched data. Can also be stored in a slower drive location. This data is cached because processing itself is slow.
* **training data**: data stored in a format optimized for training a particular model class. This should be stored at a location which allows for fast access during training, i.e. the local SSD drive.
* **experiment data** generated during experiments, e.g. losses, metric, weights and biases. For example, place WandB data under this root directory.

*Please delete all data whose license expires.*

#### Caching

Caching in is version-aware. You will note that a number of functions have a ``@version`` or ``@cache`` decorator. The version of a function often also depends on other
function versions. A change to the version will trigger a refesh of any cache affected by the respected function - so be careful, as this can be slow; also do not change the version of a function or class
without good reason as it may trigger refreshs for others once they check out.
Chaning a core function - e.g. massive's client version - will trigger a
redownload and recalculation of *everything* you have on disk. (If such hard refresh is ever required, either just delete all data or set the environment variable ``QUANTDATA_CACHE_MODE`` to ``"update"``, see also below.)

Please also check the documentation for [``@version``](https://quantitative-research.de/docs/cdxcore/api/generated/cdxcore.version.html) and [``@cache``](https://quantitative-research.de/docs/cdxcore/api/generated/cdxcore.subdir.html#cdxcore.subdir.SubDir.cache) with the package ``cdxcore``.

#### Defining Data Directories

You will need to set your directory and data access variables.
All directory definitions follow the [``cdxcore.subdir``](https://quantitative-research.de/docs/cdxcore/api/generated/cdxcore.subdir.html) notation, i.e. you can refer to the temporary directory by ``"!"`` and to your home directory by ``"~"``. Note that under Linux data stored in ``/tmp/`` are deleted upon reboot. 
If you do not need to/can/want to split data across storage locations, you can just set:

* ``QUANTDATA_ROOT``: if you have everything in one place anyway, use this to set all of the above in one step.

To optimize access speed, you can split each area by device:

* ``QUANTDATA_RAW_DATA``: root directory for storing raw cached data. Usually does not need to be fast, e.g. a network drive. Defaults to ``QUANTDATA_ROOT/raw``.
* ``QUANTDATA_PROCESSING_DATA``: intermediate processing: good for multi-process CPU access, but does not need to deliver to GPU. Defaults to ``QUANTDATA_ROOT/processing``.
* ``QUANTDATA_TRAINING_DATA``: training data with the most effective disk to GPU location.
  Defaults to ``QUANTDATA_ROOT/training``. 
* ``QUANTDATA_EXPERIMENTS``: where to store data for experiments, e.g. losses, metrics, weights and biases locally.  Defaults to ``QUANTDATA_ROOT/experiments``.

* ``QUANTDATA_TEMP_DATA``: local ephermal caching. Very useful for exceedingly slow processes. Not used at the moment. Defaults to ``!/quantdata``.

In Linux, setting the environment variables can be achieved by adding the following to ``~/.bashrc``:

```bash
# Data directories
export QUANTDATA_RAW_DATA=/path/to/data/raw
export QUANTDATA_PROCESSING_DATA=/path/to/data/processing
export QUANTDATA_TRAINING_DATA=/fast_path/to/data/training
export QUANTDATA_EXPERIMENTS=/fast_path/to/data/training
```
Alternatively, you can set a global directory. Example here to make it relative to ``$HOME``:
```bash
export QUANTDATA_ROOT=~/quantdata_data/
```

### Data Access

Each data source support by our project will require you to register the respective login data, also as environment variables. At this point the following two sources are supported:

* **[Massive.com](quantdata/massive/README.md)**.
  1. Ensure you have an account and test it.
  2. Set the ``MASSIVE_TOKEN`` environment variable to your API token.
  
* **[WRDS Option Metrics](quantdata/widb/README.md)** including **Option Metrics**
  1. [Create an account](https://wrds-www.wharton.upenn.edu/pages/support/getting-started/) and [enable MFA](https://wrds-www.wharton.upenn.edu/pages/about/log-in-to-wrds-using-two-factor-authentication/).
  2. [Follow the WRDS instructions](https://wrds-www.wharton.upenn.edu/pages/support/programming-wrds/programming-python/python-from-your-computer/) to ensure you have 
    the ``pgpass`` pass file. *This is important.*
  3. Set the ``WRDS_USERNAME`` variable to your username.   
* **[Yahoo Finance](quantdata/yahoo/README.md)**.
  1. Ensure you have an account and test it.
  2. Set the ``MASSIVE_TOKEN`` environment variable to your API token.

### Other Useful Environment Variables

* ``QUANTDATA_CACHE_MODE``: sets the global [``cdxcore.subdir.CacheMode``](https://quantitative-research.de/docs/cdxcore/api/generated/cdxcore.subdir.html#cdxcore.subdir.CacheMode).
  For example, to turn off all caching use ``QUANTDATA_CACHE_MODE="off"``. To update all cached data used, use ``QUANTDATA_CACHE_MODE="update".
* ``QUANTDATA_TEST_MODE``: global flag to tell the project we are in testing mode. In this mode, no data is downloaded from any data source.

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

Code generated docuemntation is still in progress; it can be found here [https://hansbuehler.github.io/quant_data](https://hansbuehler.github.io/quant_data/).

Please make an effort to document any files you commit. The project uses [reStructuredText](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html)
which is tedious but the standard. Here is a [comprehensive introduction](https://www.sphinx-doc.org/en/master/).

You build the documentation with
```bash
cd docs
make html
```

The reStructuredText entry point is `docs/index.rst`.
