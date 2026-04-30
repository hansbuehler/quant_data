"""
quantdata

Library for handling quantitative financial data.
"""

from __future__ import annotations

__version__ = "0.1.0"

import warnings
import multiprocessing.util as mp_util
from dataclasses import dataclass
import os as os
from pathlib import Path    

from cdxcore.subdir import SubDir, version, CacheController, CacheMode#NOQA
from cdxcore.verbose import Context#NOQA
from cdxcore.pretty import PrettyObject#NOQA
from cdxcore.err import verify_inp

# --------------------------------------------------------------
# system wide settings; import file ahead of anything else
# --------------------------------------------------------------

# remove ugly warning format
def _one_line_warning(message, category, filename, lineno, line=None):
    """ Remove ugly default warning format """
    return f"{filename}:{lineno}: {category.__name__}: {message}\n"
warnings.formatwarning = _one_line_warning

# turn on multiprocessing logging to stderr (for debugging; remove in production version)
mp_util.log_to_stderr(level=mp_util.INFO)  # TODO: remove in production version                      

# --------------------------------------------------------------
# Library environment
# --------------------------------------------------------------

@dataclass(frozen=True)
class Locations:
    """
    A general reference for locations for data sources.
    Typically, data is organized a head location:

    * :attr:`raw`: for raw data downloads, typically not fast (e.g. a NAS drive)
    * :attr:`processing`: for intermediate data products, e.g. cleaned and merged datasets. Medium storage speed, mostly CPU accessed.
    * :attr:`training`: for final training input data. This location should be fast to access for training.
    * :attr:`temp`: for temporary caching of data, e.g. from API calls. Assume this data expires after the process ends.

    Underneath these locations each data source has its own subdirectory e.g. "wrds".
    Under these, further structure is typically imposed by cached files and their parameters.

    Note that all directories are reset if the environment variable ``QUANTDATA_TEST_MODE`` is set to ``1``,
    in which case they point to a local data store in ``tests/stored_data/raw``, ``tests/stored_data/processing``, and ``tests/stored_data/training`` respectively. 
    This is used for testing.
    """
    raw: SubDir #: for raw data downloads, e.g. from WRDS (the actual WRDS locations defined in :attr:`quantdata.wrds`)
    processing: SubDir #: for intermediate data products, e.g. cleaned and merged datasets
    training: SubDir #: for final training input data
    temp: SubDir #: for temporary caching of data, e.g. from API calls. This is typically a local directory with fast read/write access, and is used for caching API calls for speed. This directory is not meant for long-term storage of data, and may be cleared at any time. Use the other locations for long-term storage of data.

    def __post_init__(self):
        verify_inp( isinstance( self.raw, SubDir), f"Expected 'raw' to be of type SubDir, found {type(self.raw)}")
        verify_inp( isinstance( self.processing, SubDir), f"Expected 'processing' to be of type SubDir, found {type(self.processing)}")
        verify_inp( isinstance( self.training, SubDir), f"Expected 'training' to be of type SubDir, found {type(self.training)}")
        verify_inp( isinstance( self.temp, SubDir), f"Expected 'temp' to be of type SubDir, found {type(self.temp)}")

    def __call__(self, name: str, *, 
                 default_fmt: SubDir.Format | None = None,
                 fmt_raw : SubDir.Format | None = None,
                 fmt_processing : SubDir.Format | None = None,
                 fmt_training : SubDir.Format | None = None,
                 fmt_temp : SubDir.Format | None = None) -> Locations:
        """
        Create locations and :attr:`cdxcore.subdir.Format` for a specific data source.

        The default inherits the format of the parent. The root format is ``PICKLE``
        which always works but is not efficient.
            
        * If the function you are caching returns a pandas DataFrame, use ``SubDir.PANDAS_PARQUET`` for better performance.
        * For polars DataFrames, consider setting the default format to ``SubDir.POLARS_PARQUET``.
        * The format ``SubDir.PYTREE_HDF5`` is also supported, and is the fastest for read i/o, but `the format has limitations
          in supported output data <https://docs.h5py.org/en/stable/index.html>`_. Best used for ``training`` data.

        Parameters
        ----------
        name: str
            The name of the data source or product, e.g. "wrds" or "crsp".
        default_fmt : SubDir.Format | None, default ``None``
            A :attr:`cdxcore.subdir.Format` default format for the data.
        fmt_raw : SubDir.Format | None, default ``None``
            A :attr:`cdxcore.subdir.Format` format for the raw data. If ``None``, use ``default_fmt``.
        fmt_processing : SubDir.Format | None, default ``None``
            A :attr:`cdxcore.subdir.Format` format for the processing data. If ``None``, use ``default_fmt``.
        fmt_training : SubDir.Format | None, default ``None``
            A :attr:`cdxcore.subdir.Format` format for the training data. If ``None``, use ``default_fmt``.
        fmt_temp : SubDir.Format | None, default ``None``
            A :attr:`cdxcore.subdir.Format` format for the temporary data. If  ``None``, use ``default_fmt``.
        """
        return Locations(
            raw = self.raw(name, fmt=default_fmt if fmt_raw is None else fmt_raw),
            processing = self.processing(name, fmt=default_fmt if fmt_processing is None else fmt_processing),
            training = self.training(name, fmt=default_fmt if fmt_training is None else fmt_training),
            temp = self.temp(name, fmt=default_fmt if fmt_temp is None else fmt_temp)
        )

class Environment(object):
    """
    Defines the operating environment.

    This class is a "singleton": access it via the global variable :attr:`quantdata.env`.
    Key attributes:

    * :attr:`quantdata.Environment.locations`: a hierarchy of :class:`cdxcore.subdir.SubDir` directories for data caching:
      * :attr:`locations.raw`: for raw data downloads, e.g. from WRDS (the actual WRDS locations defined in :attr:`quantdata.wrds`)
      * :attr:`locations.processing`: for intermediate data products, e.g. cleaned and merged datasets
      * :attr:`locations.training`: for final training input data
      * :attr:`locations.temp`: for temporary caching of data, e.g. from API calls. This is typically a local directory with fast read/write access, and is used for caching API calls for speed. This directory is not meant for long-term storage of data, and may be cleared at any time. Use the other locations for long-term storage of data.
   
    * :attr:`quantdata.Environment.test_mode` is a boolean which indicates whether the code is running in test mode.
      In test mode, any attempt to connect to WRDS will raise an error, and the code should rely on cached data in ``tests/stored_data/raw/`` instead.
      To enable test mode, set the environment variable ``QUANTDATA_TEST_MODE`` to ``1``.

    Debugging
    ^^^^^^^^^
    The package uses caching throughout which can obfuscate errors if the error is in some of the cached files.
    To turn off caching globally, set the environment variable ``QUANTDATA_CACHE_MODE`` to ``"off"``.
    This will force the code to always fetch fresh data and not use any cached files.

    Developer Comment
    ^^^^^^^^^^^^^^^^^
    To generate test cases for offline testing, data sourcing will be redirected to ``tests/stored_data/``.
    To generate tests in for this directory, you can use the protected member ``_test_data_locations``.
    """

    CACHE_EXCLUDE_ARGS = ["self", Context, "WRDSClient", "MassiveClient"]

    def __init__(self):
        # identify directories
        # ---------------------
        self.test_mode  : bool = os.environ.get("QUANTDATA_TEST_MODE", "0") == "1"  # turn off any connections

        cache_mode = os.environ.get("QUANTDATA_CACHE_MODE", None)
        cc = CacheController(exclude_arg_types=self.CACHE_EXCLUDE_ARGS, cache_mode=cache_mode)

        # where tests are stored.   
        # in non-test mode, use these for generation

        def generate_test_location():
            cc = CacheController(exclude_arg_types=self.CACHE_EXCLUDE_ARGS, cache_mode="on")
            root = SubDir( str(Path(__file__).parent.parent / "tests" / "stored_data"), create_directory=False ) # directory for generating tests.
            return Locations(
                raw = root("raw", create_directory=False),
                processing= root("processing", create_directory=False),
                training = root("training", create_directory=False),
                temp = root("!/quantdata", create_directory=False)
            )
        self._test_data_locations = generate_test_location() #: Locations for test data. Used for developers only.

        # generate environment
        # test or not

        if not self.test_mode:
            raw        = os.environ.get("QUANTDATA_RAW_DATA", None)
            processing = os.environ.get("QUANTDATA_PROCESSING_DATA", None)
            training   = os.environ.get("QUANTDATA_TRAINING_DATA", None)
            def_root   = os.environ.get("QUANTDATA_ROOT", None)
            temp       = os.environ.get("QUANTDATA_TEMP_DATA", "!/quantdata")

            if not cache_mode is None:
                try:
                    cache_mode = CacheMode(cache_mode)
                except ValueError as e:
                    raise ValueError(f"Invalid cache mode specified in environment variable QUANTDATA_CACHE_MODE: '{cache_mode}': {e}.") from e
                warnings.warn(f"Using cache mode '{cache_mode}' as specified in environment variable QUANTDATA_CACHE_MODE.")

            raw        = SubDir(raw) if not raw is None else None
            processing = SubDir(processing) if not processing is None else None
            training   = SubDir(training) if not training is None else None
            temp       = SubDir(temp) if not temp is None else None

            if raw is None or processing is None or training is None:
                if def_root is None:
                    raise RuntimeError(
                                    "Could not identify all working directories: use the following environment variables: "+\
                                    f"QUANTDATA_RAW_DATA: '{'(missing)' if raw is None else raw }'; "+\
                                    f"QUANTDATA_PROCESSING_DATA: '{'(missing)' if processing is None else processing }'; "+\
                                    f"QUANTDATA_TRAINING_DATA: '{'(missing)' if training is None else training }'; "+\
                                    f"QUANTDATA_TEMP_DATA: '{'(missing)' if temp is None else temp }'. "+
                                    "Alternatively you may set a generic QUANTDATA_ROOT directory")
                                    
                def_root = SubDir(def_root)
                raw = def_root("raw") if raw is None else raw
                processing = def_root("processing") if processing is None else processing
                training = def_root("training") if training is None else training
                temp = def_root("temp") if temp is None else temp

            raw = SubDir(raw, cache_controller=cc)
            processing = SubDir(processing, cache_controller=cc)
            training = SubDir(training, cache_controller=cc)
            temp = SubDir(temp, cache_controller=cc)

        else:
            # test mode
            # ---------
            # In test mode, we use locally stored data.
            # We also do not write to the cache (which is now pointing to a local data store)
            test_cc    = CacheController(exclude_arg_types=self.CACHE_EXCLUDE_ARGS, cache_mode="cacheonly")

            raw        = self._test_data_locations.raw(cache_controller=test_cc)
            training   = self._test_data_locations.training(cache_controller=test_cc)
            processing = self._test_data_locations.processing(cache_controller=test_cc)
            # temp is different: will create a directory which self-deletes upon deletion of the object.
            temp       = SubDir("?/quantdata_test_temp", cache_controller=test_cc, create_directory=True) # this can always read/write. 

            if not raw.path_exists():
                raise FileNotFoundError(f"Could not find stored raw data for testing at: {raw.path}")
            if not training.path_exists():
                raise FileNotFoundError(f"Could not find stored training data for testing at: {training.path}")
            if not processing.path_exists():
                raise FileNotFoundError(f"Could not find stored processing data for testing at: {processing.path}")
            
        self.locations = Locations(
            raw = raw,
            processing = processing,
            training = training,
            temp = temp
        )

    def generate_raw_test_targe():
        pass

env = Environment()
""" One global instande of the :class:`quantdata.Environment` class, which defines the operating environment. """

