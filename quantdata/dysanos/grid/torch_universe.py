"""
SANOS (Torch)
===========

Fitting the DLVs of SANOS.
"""

from __future__ import annotations

from collections.abc import Callable
import os
import re
import warnings as warnings

from cdxcore.subdir import SubDir
from cdxcore.uniquehash import named_unique_filename48_8
import numpy as np
from dataclasses import asdict, dataclass, field
from cdxcore.err import error, verify, verify_inp
from cdxcore.verbose import Context
from cdxcore.version import version
from cdxcore.bs import bs
from cdxcore.pretty import PrettyObject
from cdxcore.util import fmt_big_byte_number, fmt_digits
import math
import torch

from quantdata.biz_calendar import DAYS_PER_YEAR
from quantdata.dysanos.grid.torch_simulator import ARSampler
from quantdata.dysanos.grid.torch_model import SanosModel

if os.environ.get("DYSANONS_USE_LAPTOP") is not None:
    universe_cache = SubDir(r"C:\users\public\.dysanos_universe/", fmt=SubDir.PYTREE_HDF5)
else:
    universe_cache = SubDir("~/.dysanos_universe", fmt=SubDir.PYTREE_HDF5)

def _uid( model : SanosModel,
         sampler : ARSampler,
         market : PrettyObject, 
         num_steps : int, 
         num_samples : int,
         bandwidth : float, 
         dt : float ) -> str:
    """ Create a unique identifier for the universe based on the model, sampler, market, and parameters. """
    a = named_unique_filename48_8(str(model.config.group))
    b = named_unique_filename48_8(str(model.label))
    c = named_unique_filename48_8(f"{num_steps} {num_samples} {bandwidth} {dt}", num_steps, num_samples, bandwidth, dt)
    d = named_unique_filename48_8(str(sampler.config), sampler.config)
    e = str(sampler.rng_current_state_uid)  # this is essentially the seed
    return f"{a}/{b}/{c}/{d}/{e}"

@universe_cache.cache("0.2", dependencies=[ARSampler, SanosModel.create_paths],uid=_uid )
def create_universe(    model : SanosModel,
                        sampler : ARSampler,
                        market : PrettyObject, 
                        reset_rng : bool, *, 
                        num_steps : int = 100, 
                        num_samples : int = 1,
                        bandwidth : float = 0.2, 
                        dt : float = 1./DAYS_PER_YEAR, 
                        block_size : int|None = 1000, 
                        override_cache_mode : str|None = None,
                        verbose : Context = Context.all ) -> dict[str, np.ndarray]:
    """ 
    Create universe of calls and vols from a sampler and market definition. 
    This function is cached, so that repeated calls with the same parameters will return the same universe without recomputing it.
    """

    if reset_rng:
        sampler.reset_rng()

    if not block_size is None and num_samples > block_size:
        num_blocks = math.ceil(num_samples / block_size)
        tme = verbose.timer()

        EXCLUDE = {'dt', 'bandwidth','expiries'}

        universe = None
        for i in range(num_blocks):
            verbose.write(lambda : f"\rCreating universe for {model.qualified_label} of {fmt_digits(num_samples)} samples over {num_steps} time steps with a dt of {dt*DAYS_PER_YEAR:.2f}d and a bandwidth of {bandwidth:.3g} with a blocksize of {block_size}: block {i+1}/{num_blocks}; so far this took {tme}... ", end='')
            start = i * block_size
            end = min((i + 1) * block_size, num_samples)
            r = create_universe(        model=model,
                                        sampler=sampler, 
                                        market=market, 
                                        reset_rng=False, 
                                        num_steps=num_steps, 
                                        num_samples=end-start, 
                                        block_size=None,
                                        bandwidth=bandwidth, 
                                        dt=dt,
                                        override_cache_mode=override_cache_mode,
                                        verbose=Context.quiet )
    

            if universe is None:
                universe = { k: [v] for k,v in r.items() if not k in EXCLUDE }
            else:
                for k,v in r.items():
                    if k not in EXCLUDE:
                        assert isinstance(v, np.ndarray), f"Expected numpy array for {k}, got {type(v)}"
                        universe[k].append(v)
        # concatenate results
        universe = PrettyObject({ k: np.concatenate(v, axis=0) for k,v in universe.items() })
        size    = sum( v.nbytes for v in universe.values() if not v is None )
        verbose.write(lambda : f"\rCreated universe for {model.qualified_label} of {num_samples} samples over {num_steps} time steps in {num_blocks} blocks of size {block_size} a dt of {dt*DAYS_PER_YEAR:.2f}d and a bandwidth of {bandwidth:.3g}: this took {tme}. Memory usage: {fmt_big_byte_number(size)} bytes, which is {fmt_big_byte_number(int(size/num_samples))} bytes per sample.")

    else:
        with torch.no_grad():
            with verbose.write_t(lambda : f"Generating {num_samples} samples of {model.qualified_label} for {num_steps} steps... ", end='') as tme:
                xstate  = sampler.free_sample_xstates(  num_steps=num_steps, 
                                                        num_samples=num_samples, 
                                                        dt = dt,
                                                        bandwidth = bandwidth,
                                                        )
                spots             = np.exp(xstate[...,0])
                state             = xstate[...,1:]
                assert spots.shape == (num_samples, num_steps,), f"Spot dimension error {spots.shape} != {(num_samples, num_steps)}"
                assert state.shape == (num_samples, num_steps, model.num_state), f"State dimension error {state.shape} != {(num_samples, num_steps, model.num_state)}"
                
                spots             = torch.tensor( spots, dtype=model.env.fdtype, device=model.env.device )
                expiries          = torch.tensor( market.grid_market_def.grid_config.grid_expiries, dtype=model.env.fdtype, device=model.env.device )
                state             = torch.tensor( state, dtype=model.env.fdtype, device=model.env.device )

                verbose.write(f"generated state of size {fmt_big_byte_number(spots.nbytes+state.nbytes)}; now building market:", head=False)
                universe          = model.create_paths( spots, 
                                                        expiries=expiries, 
                                                        dt=dt,
                                                        state=state, with_vols=True,
                                                        #block_size = None,
                                                        verbose = verbose(1) )
                size               = sum( v.nbytes for v in universe.values() if not v is None )
                universe = PrettyObject({ k: v.detach().cpu().numpy() for k,v in universe.items() })  # TODO fix this - at the moment required for caching
                universe['xstate'] = xstate
                verbose.write(lambda : f"Generated universe; this took {tme}. Using {fmt_big_byte_number(size)} equal to {fmt_big_byte_number(int(size/num_samples))} per sample.")

    # add utilities
    universe.dt = dt
    universe.bandwidth = bandwidth 
    universe.expiries = market.grid_market_def.grid_config.grid_expiries
    return universe
