from quantdata.dysanos import env, SubDir, version
from quantdata.widb.client import WIDBClient, SubDir

_training_root   = WIDBClient.locations.training("dysanos/data",fmt=SubDir.PANDAS_PARQUET) 
""" Root cache for training data """
