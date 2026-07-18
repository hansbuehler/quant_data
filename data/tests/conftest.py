from __future__ import annotations

import os
from pathlib import Path


stored_data_root = Path(__file__).resolve().parent / "stored_data"

for directory_name in ("raw", "processing", "training", "experiments"):
	(stored_data_root / directory_name).mkdir(parents=True, exist_ok=True)

os.environ.setdefault("QUANTDATA_TEST_MODE", "1")
os.environ.setdefault("QUANTDATA_TEST_DATA", str(stored_data_root))
