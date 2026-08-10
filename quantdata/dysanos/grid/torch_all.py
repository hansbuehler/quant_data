"""
Whole training setup
"""

from dataclasses import dataclass, asdict
import gc
from cdxcore.pretty import PrettyObject
import numpy as np

from quantdata.dysanos.grid.torch_simulator import plot_sampler
import os
from cdxcore.subdir import SubDir
from cdxcore.uniquehash import named_unique_filename48_8

if os.environ.get("DYSANONS_USE_LAPTOP") is not None:
    universe_cache = SubDir(r"C:\users\public\.dysanos_all/", fmt=SubDir.PYTREE_HDF5)
else:
    universe_cache = SubDir("~/.dysanos_all", fmt=SubDir.PYTREE_HDF5)

@dataclass(frozen=True)
class DysanosExperiment:
    grid_strikes : int = 100
    num_alpha    : int = 5
    num_steps    : int = 10
    num_samples  : int = 25_000
    pcnt_validation : float = 0.1
    clip_grad_norm : float|None = None

    def __str__(self):
        return f"{self.num_samples},steps={self.num_steps},gs={self.grid_strikes},alpha={self.num_alpha},pcnt_validation={self.pcnt_validation},clip_grad_norm={self.clip_grad_norm}"

def run_experiment(
        experiment : DysanosExperiment, *,
        plot_sampler : bool = True,
        plot_det_measure_change : bool = True
):
    gc.collect()
    print("Importing torch...", end='', flush=True)
    import torch
    print("done.")

    # load market data
    # ----------------

    gc.collect()
    from quantdata.dysanos.grid.torch_market import GridMarketDefinition, DateRange, PoolConfig, Context, date, GridConfig

    print("** Starting to load market data...")
    grid_config     = GridConfig(grid_num_strikes=experiment.grid_strikes)
    grid_market_def = GridMarketDefinition( ticker="SPX", date_range=DateRange(start=date(2020, 1, 1), end=date(2025, 8, 30)), grid_config=grid_config )
    market          = grid_market_def.load( pool_config = PoolConfig(num_workers=4, threading=False ), override_cache_mode="cacheonly", verbose=Context.all )
    print(f"nms /done - found {len(market['dates'])} dates.")

    # load trained model
    # ------------------

    from quantdata.pool import PoolConfig
    from quantdata.dysanos.grid.torch_gfx import VizConfig
    from quantdata.dysanos.grid.torch_train import train, TrainConfig
    from quantdata.dysanos.grid.torch_model import BaseEnvironment, FeedForward, SanosModelConfig, SanosDecoderConfig

    mdl_config = SanosModelConfig( 
        label="ac_state20",
        user_version="0.2",
        group="sm25_100",
        encoder_config = FeedForward.Config(
                activation = "SELU",
                num_depth = 2,
                num_width = 100,
                use_bias = True,
            ),
        decoder_config = SanosDecoderConfig(
            smoothness = 0.25,
            num_state = 20, 
            nn_config = FeedForward.Config(
                activation = "SELU",
                num_depth = 2,
                num_width = 100,
                use_bias = True,
            ),
        ),
    )

    model        = None
    torch_data   = None
    env          = BaseEnvironment(device="cpu", debug=False)
    viz_config   = VizConfig( num_viz_dates=4 )
    trn_config   = TrainConfig( num_epochs=100_000, lr=1E-4, final_lr=1E-6, weight_decay=1e-4, batch_size=400, compile=True, epoch_interval=10000, clip_grad_norm=experiment.clip_grad_norm )
    print("Loading trained model...")
    model_output,\
    model,\
    torch_data   = train( market, mdl_config=mdl_config, trn_config=trn_config, viz_config=viz_config, just_load="plot", env=env )
    torch_data   = mdl_config.torch_data( env = env, market = market, verbose = Context.quiet ) if torch_data is None else torch_data
    del model_output, mdl_config, viz_config, env
    gc.collect()
    print("Trained model loaded.")

    # AR(1) PCA
    # --------=-

    from quantdata.dysanos.grid.torch_simulator import ARSamplerConfig, ARSampler, Context, plot_sampler as plot_ar_sampler

    print("Computing AR1(1) PCA")
    ar_config      = ARSamplerConfig(num_alpha=experiment.num_alpha, use_matrix_exp=False)
    sampler        = ARSampler( market, model, torch_data, config=ar_config, verbose=Context.all )
    gc.collect()
    print("AR(1) PCA computed:")
    sampler.report_system()

    if plot_sampler:
        plot_ar_sampler(sampler)

    # simulate universe
    # -----------------

    from quantdata.dysanos.grid.torch_universe import create_universe

    print("Simulating universe")
    universe = create_universe( model=model, sampler=sampler, market=market, reset_rng=True,
                               num_steps=experiment.num_steps,
                               num_samples=experiment.num_samples,
                               block_size=100000,
                               bandwidth=0.2)
    gc.collect()
    print("Simulated universe")
    for k,v in universe.items():
        if v is None:
            continue
        if isinstance(v, np.ndarray):
            print( k, v.shape, v.nbytes, v.dtype, "ok" if np.all(np.isfinite(v)) else f"** {np.sum(np.isfinite(v))}")
        else:
            print( k, v )

    # deterministic measure change
    # ----------------------------

    from quantdata.dysanos.grid.torch_det_measure_change import change
    dQdP_dS = change(universe, plot=plot_det_measure_change )
    gc.collect()

    return PrettyObject(
        market=market,
        model=model,
        dQdP_dS=dQdP_dS,
        universe=universe,
        sampler=sampler
    )

@universe_cache.cache("0.1", uid =lambda experiment : named_unique_filename48_8(f"tight_{experiment}", experiment))
def run_experiment_tight(
        experiment : DysanosExperiment, *,
        plot_sampler : bool = True,
        plot_det_measure_change : bool = True
):
    ex = run_experiment(experiment, plot_sampler=plot_sampler, plot_det_measure_change=plot_det_measure_change)
    return dict( universe=ex.universe, dQdP_dS=ex.dQdP_dS )


@universe_cache.cache("0.1", uid =lambda experiment : named_unique_filename48_8(f"tight_{experiment}", experiment))
def run_experiment_triple(
        experiment : DysanosExperiment, *,
        plot_sampler : bool = True,
        plot_det_measure_change : bool = True
):
    ex = run_experiment(experiment, plot_sampler=plot_sampler, plot_det_measure_change=plot_det_measure_change)
    return dict( universe=ex.universe, dQdP_dS=ex.dQdP_dS )

