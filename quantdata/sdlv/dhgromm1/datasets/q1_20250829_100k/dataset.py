"""Portable NumPy-only loader for the deliberately retained GROMM Q1 dataset.

Run this file directly; importing quantdata, Torch, JAX, or host configuration
is not required. Large chunks are fetched by Git LFS, not regular Git blobs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterator

import numpy as np

SCHEMA = "dhgromm1-share-q1-training-v1"
ARRAYS = {"prices":((21,21),"float64"),"gains":((20,21),"float64"),
          "payoff":((),"float64"),"changes_packed":((3990,),"float32")}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()


def expand_changes(packed: np.ndarray, *, day: int | None = None) -> np.ndarray:
    """Cash mark changes of earlier cohorts, with zero padding for unopened ones.

    Entry [path,t,21*r+i] is mark_t - opening_mark for instrument i opened on
    day r<t, frozen after settlement. Multiply by actual units a[r,i]/S_r,
    not by a[r,i]/option_premium. Day-zero marks and r>=t entries are zero.
    """
    if packed.ndim!=2 or packed.shape[1]!=3990 or packed.dtype!=np.float32:
        raise ValueError("expected float32 packed marks with shape (paths,3990)")
    if day is not None and (not isinstance(day,int) or not 0<=day<20):
        raise ValueError("decision day must be an integer in [0,19]")
    days = range(20) if day is None else (day,)
    result = np.zeros((len(packed),20 if day is None else 1,420),dtype=np.float32)
    for slot,t in enumerate(days):
        begin,end = 21*t*(t-1)//2,21*t*(t+1)//2
        result[:,slot,:21*t] = packed[:,begin:end]
    return result if day is None else result[:,0]


def verify_arrays(arrays: dict[str,np.ndarray], *, paths: int) -> None:
    if set(arrays)!=set(ARRAYS):
        raise ValueError("unexpected chunk array names")
    for name,(tail,dtype) in ARRAYS.items():
        value = arrays[name]
        if value.shape!=(paths,*tail) or value.dtype!=np.dtype(dtype) or not np.isfinite(value).all():
            raise ValueError(f"invalid {name} shape/dtype/finiteness")
    p,r,h = (arrays[k] for k in ("prices","gains","payoff"))
    if paths<2 or paths%2 or np.any(p[:,:,0]<=0):
        raise ValueError("expected complete antithetic pairs and positive spots")
    np.testing.assert_allclose(h,np.abs(p[:,20,0]/p[:,10,0]-1),atol=2e-12,rtol=1e-12)
    np.testing.assert_allclose(r[:,:,0],p[:,20:21,0]/p[:,:20,0]-1,atol=2e-12,rtol=1e-12)
    np.testing.assert_allclose(r[:,10,1],np.maximum(p[:,20,0]/p[:,10,0]-1,0)-p[:,10,1]/p[:,10,0],atol=2e-12,rtol=1e-12)
    np.testing.assert_allclose(r[:,10,17],p[:,20,0]/p[:,10,0]-p[:,10,17]/p[:,10,0],atol=2e-12,rtol=1e-12)


class Dataset:
    """Stream complete pairs in original order, checking SHA-256 by default."""

    def __init__(self, root: Path | str | None = None, *, verify_hashes: bool = True) -> None:
        self.root = (Path(__file__).parent if root is None else Path(root)).resolve()
        self.manifest = json.loads((self.root/"MANIFEST.json").read_text())
        if self.manifest.get("schema")!=SCHEMA:
            raise ValueError("unsupported sharing dataset version")
        self.verify_hashes = verify_hashes
        completed = 0
        for record in self.manifest["chunks"]:
            if record["start"]!=completed or record["count"]%2 or record["count"]<2:
                raise ValueError("chunk order must preserve complete original antithetic pairs")
            self._path(record["file"])
            completed += record["count"]
        if completed!=self.manifest["paths"] or completed%2:
            raise ValueError("incomplete dataset population")

    def _path(self, name: str) -> Path:
        path = (self.root/name).resolve()
        if Path(name).is_absolute() or not path.is_relative_to(self.root):
            raise ValueError("manifest file escaped dataset directory")
        return path

    def _check(self, record: dict) -> Path:
        path = self._path(record["file"])
        if self.verify_hashes and sha256(path)!=record["sha256"]:
            raise ValueError(f"data checksum mismatch: {path.name}; run git lfs pull if this is a pointer")
        return path

    def normalization(self) -> dict[str,np.ndarray]:
        with np.load(self._check(self.manifest["normalization"]),allow_pickle=False) as saved:
            return {k:saved[k] for k in saved.files}

    def chunks(self) -> Iterator[tuple[dict,dict[str,np.ndarray]]]:
        for record in self.manifest["chunks"]:
            with np.load(self._check(record),allow_pickle=False) as saved:
                arrays = {k:saved[k] for k in saved.files}
            verify_arrays(arrays,paths=record["count"])
            yield record,arrays

    def summary(self) -> dict:
        """Validate every chunk with bounded memory and report paired payoff MC SE."""
        pairs,total,total_square = 0,0.,0.
        gain_sum = np.zeros((20,21),dtype=np.float64)
        for record,arrays in self.chunks():
            sample = arrays["payoff"].reshape(-1,2).mean(1)
            pairs += len(sample)
            total += float(sample.sum())
            total_square += float(np.square(sample).sum())
            gain_sum += arrays["gains"].sum(0)
            print(f"verified {record['start']+record['count']}/{self.manifest['paths']} paths",flush=True)
        normal = self.normalization()
        np.testing.assert_allclose(gain_sum/self.manifest["paths"],normal["return_mean"],atol=1e-12,rtol=1e-10)
        mean = total/pairs
        se = np.sqrt(max((total_square-pairs*mean*mean)/(pairs-1),0)/pairs)
        return dict(paths=2*pairs,independent_antithetic_pairs=pairs,mean_payoff=mean,three_mc_standard_errors=3*se,
                    schema=self.manifest["schema"],status="verified")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path(__file__).parent)
    parser.add_argument("--verify",action="store_true",help="check all chunks, hashes and payoff/replication identities")
    args = parser.parse_args()
    data = Dataset(args.root)
    print(json.dumps(data.summary() if args.verify else data.manifest,indent=2),flush=True)


if __name__=="__main__":
    main()
