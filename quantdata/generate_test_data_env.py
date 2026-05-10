"""
Import this file when generating data for testing.
Importing this file will set the default cache directories to 
the correct path for``"tests/stored_data"``.
"""

from __future__ import annotations
from cdxcore.subdir import SubDir
from cdxcore.util import fmt_dict
from dataclasses import asdict
import warnings
import os
def _set_raw_testing_dir():
    if 'QUANTDATA_ENVIRONMENT_LIVE' in globals():
        raise RuntimeError("Error importing 'utils/generate_test_data_env.py': QUANTDATA_ENVIRONMENT_LIVE is already set. "+\
                           "It file must be imported ahead of any other quantdata imports." +\
                           "This file should only be imported when generating test data, not in live code or tests.")

    cwd  = os.getcwd()
    root = "quant_data"
    ix   = cwd.find(root)
    assert ix!=-1, f"Error importing 'utils/generate_test_data_env.py': current working directory must contain '{root}'. Got cwd={cwd}"
    os.environ.pop('QUANTDATA_RAW_DATA', None)
    os.environ.pop('QUANTDATA_PROCESSING_DATA', None)
    os.environ.pop('QUANTDATA_TRAINING_DATA', None)
    os.environ.pop('QUANTDATA_EXPERIMENTS', None)
    os.environ.pop('QUANTDATA_TEST_MODE', None)
    os.environ["QUANTDATA_ROOT"] = SubDir(cwd[:ix+len(root)])("tests")("stored_data").path
    os.environ["QUANTDATA_CACHE_MODE"] = "update"

    warnings.warn("\n** WARNING ***:\n'utils/generate_test_data_env.py' was imported. This import is used to generate test data.\n"+\
                 "QUANTDATA_ROOT now points to " + os.environ["QUANTDATA_ROOT"] + " and QUANTDATA_CACHE_MODE is set to 'update'.")

    def impt():
        from quantdata import env
        warnings.warn(f"Locations are set: {fmt_dict(asdict(env.locations))}")
    impt()

_set_raw_testing_dir()
