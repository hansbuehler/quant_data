from quantdata.dysanos import env, SubDir, version
from quantdata.torch import BaseModel, BaseEnvironment

_training_root   = env.locations.training("dysanos/train",fmt=SubDir.PANDAS_PARQUET) 
""" Root cache for training data """
